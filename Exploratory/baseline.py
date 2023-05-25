# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 0. Foreward

# + Task1: How many contents do each topic have?

# + Task2: Which contents do each topic have?


# + Task1: How many contents do each topic have? [markdown]
# # 1. Preparation
# + Task2: Which contents do each topic have? [markdown]
# ## 1.1. Load Libraries & Settings

# +
import pandas as pd
import numpy as np
import warnings
import torch
from torch import nn
from torch.nn import functional as F
import plotly.express as px
import pickle
import os
import gc
import re
import random
import lightgbm as lgb
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as tts

warnings.simplefilter("ignore")
pd.options.display.max_columns=100
pd.options.display.max_rows=100
torch.autograd.set_detect_anomaly(True)

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True

device = "cuda" if torch.cuda.is_available() else "cpu"
# %env TOKENIZERS_PARALLELISM=true
# %env CUDA_LAUNCH_BLOCKING=1
# -

class CFG:
    INPUT = "../data/"
    MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"#"sbert-models/paraphrase-multilingual-mpnet-base-v2"
    MAX_LEN = 512
    BATCH_SIZE = 2**7
    EPOCHS = 50
    THRESHOLD = .5
    EARLY_STOPPING_ROUNDS = 5


# ## 1.2. Fetching & Preprocessing Data

# ### 1.2.1. Fetching CSVs & LabelEncoding

# + tags=[]
# %%time

content = pd.read_csv(f'{CFG.INPUT}/content.csv')
correlations = pd.read_csv(f'{CFG.INPUT}/correlations.csv')
topics = pd.read_csv(f'{CFG.INPUT}/topics.csv')
sample_submission = pd.read_csv(f'{CFG.INPUT}/sample_submission.csv')

t_le = LabelEncoder()
topics_all = list(set(correlations.topic_id) | set(topics.id) | set(topics.parent)) + ["None"]
t_le.fit(topics_all)
topics.id = t_le.transform(topics.id)
topics.parent = t_le.transform(topics.parent.fillna("None"))
correlations.topic_id = t_le.transform(correlations.topic_id)

c_le = LabelEncoder()
content.id = c_le.fit_transform(content.id)

target_topics = sample_submission.topic_id
target_topics_enc = t_le.transform(target_topics)

del topics_all
gc.collect()

display(content.head())
display(correlations.head())
display(topics.head())
display(sample_submission.head())
# -

# ### 1.2.2. Preprocessing Tables

# +
# %%time

"""
correlations
"""
correlations_transform = []
for i, row in correlations.iterrows():
    for c in row.content_ids.split(" "):
        correlations_transform.append((row.topic_id, c))

correlations_transform = pd.DataFrame(
    correlations_transform, columns=["topic_id", "content_id"]
)
correlations_transform.content_id = c_le.transform(correlations_transform.content_id)

correlations_transform = pd.merge(
    correlations_transform,
    content[["id", "kind", "language", "copyright_holder", "license"]].rename(
        columns={col:"c_"+col if col != "id" else "content_id" for col in content.columns}
    ),
    how="left",
    on="content_id",
)

correlations_transform = pd.merge(
    correlations_transform,
    topics[["id", "category",	"level", "language", "has_content"]].rename(
        columns={col:"t_"+col if col != "id" else "topic_id" for col in topics.columns}
    ),
    how="left",
    on="topic_id",
)

display(correlations_transform.head())


"""
content
"""
text_cols = ["title", "description", "text"]
content_rev = content.drop(text_cols, axis=1)
content_rev["text"] = content.title.fillna("") + \
    "\n" + content.description.fillna("") + \
    "\n" + content.text.fillna("")

display(content_rev.head())

"""
topics
"""
text_cols = ["title", "description"]
agg = correlations_transform.groupby(["topic_id"]).agg(
    num_contents=("content_id", "count")
).fillna(0).reset_index().rename(columns={"topic_id":"id"})
topics_rev = pd.merge(topics, agg, how="left", on="id").drop(text_cols, axis=1).sort_index()
topics_rev.num_contents = topics_rev.num_contents.fillna(0)
topics_rev["text"] = topics.title.fillna("") + "\n" + topics.description.fillna("")

display(topics_rev.head())

del correlations, agg, content, topics
gc.collect()
# -

# ## 1.3. Feature Engineering

# ### 1.3.1. Text Embedding

# +
# %%time

model = AutoModel.from_pretrained(CFG.MODEL)
tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL)
model.eval();
model.to(device);

reprocessing = False

def distributed_representations(data):
    distributed_representations = []

    for text in data:
        tok = tokenizer(text, max_length=CFG.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            embedding = model(tok.input_ids.to(device), tok.attention_mask.to(device)).last_hidden_state
        embedding = torch.mean(embedding, dim=1)[0].cpu()
        distributed_representations += [embedding]
    distributed_representations = torch.stack(distributed_representations)
    return distributed_representations

if not os.path.exists(f"{CFG.INPUT}/dr_topics.pickle") or reprocessing:
    # %time dr_topics = distributed_representations(topics_rev.text) # Wall time: 7min 27s
    with open(f"{CFG.INPUT}/dr_topics.pickle", 'wb') as p:
        pickle.dump(dr_topics, p)
else:
    with open(f"{CFG.INPUT}/dr_topics.pickle", 'rb') as f:
        dr_topics = pickle.load(f)

if not os.path.exists(f"{CFG.INPUT}/dr_content.pickle") or reprocessing:
    # %time dr_content = distributed_representations(content_rev.text) # Wall time: 44min 49s
    with open(f"{CFG.INPUT}/dr_content.pickle", 'wb') as p:
        pickle.dump(dr_content, p)
else:
    with open(f"{CFG.INPUT}/dr_content.pickle", 'rb') as f:
        dr_content = pickle.load(f)
        
del model, tokenizer
if device == "cuda":
    torch.cuda.empty_cache()
gc.collect()

display(dr_topics)
display(dr_content)
# -

# ### 1.3.2. Analyze Channels

channel = topics_rev.groupby("channel").agg(ch_num_topics=("id", "count"))
channel = channel.join(topics_rev.groupby("channel").agg(ch_variety_lang=("language", "nunique")))
channel = channel.join(topics_rev.groupby("channel").agg(ch_max_level=("level", "max")))
channel = channel.join(topics_rev.groupby("channel").agg(ch_mean_level=("level", "mean")))
channel = channel.join(topics_rev.groupby("channel").agg(ch_std_level=("level", "std")))
channel = channel.reset_index()
channel.head()

# ### 1.3.3. Contents: Copyright & Text Processing

# +
# %%time

new_copyright = []
for ch in content_rev.copyright_holder.fillna(""):
    res = re.search(r"\s\d{4}", ch)
    if res:
        holder = ch[:res.span()[0]+1] + ch[res.span()[1]:]
        year = res.group().strip()
        new_copyright.append((holder.strip(), year))
    else:
        if ch == "":
            ch = None
        new_copyright.append((ch, np.nan))

content_rev[["copyright_holder", "copyright_year"]] = pd.DataFrame(new_copyright)
content_rev["copyright_year"] = content_rev["copyright_year"].astype(float)

del new_copyright
gc.collect()

display(content_rev.head())
# -

# ### 1.3.4. Numbers of words in text

# +
# %%time

content_rev["words_text"] = [len(re.split(r"\s", t)) for t in content_rev.text]
topics_rev["words_text"] = [len(re.split(r"\s", t)) for t in topics_rev.text]

display(content_rev.head())
display(topics_rev.head())
# -

# ### 1.3.5. Finalize Feature Enginerring

# +
task1y = topics_rev.num_contents

target_or_not = topics_rev.id.isin(target_topics_enc)
target_lang = topics_rev[target_or_not].language

topics_rev = pd.merge(topics_rev, channel, how="left", on="channel")
topics_rev = topics_rev.drop(["channel", "num_contents", "text", "parent"], axis=1)
topics_rev = pd.get_dummies(topics_rev, columns=["category", "language"], dummy_na=True).fillna(0)

topics_only_target = topics_rev[target_or_not].set_index("id")
topics_rev = topics_rev[~target_or_not].set_index("id")
task1y = task1y[~target_or_not]

has_content = topics_rev["has_content"].tolist()
topics_rev = topics_rev.drop("has_content", axis=1)

dr_topics_only_target = dr_topics[target_or_not]
dr_topics = dr_topics[~target_or_not]

lang_contentid = content_rev[["id", "language"]]
content_rev = content_rev.drop(["text"], axis=1).set_index("id").fillna(0)
content_rev = pd.get_dummies(content_rev, columns=["kind", "language", "copyright_holder", "license"], dummy_na=True)

dr_content_id = content_rev.index
dr_topics_id = topics_rev.index

del channel
gc.collect()
# -

content_rev.columns.tolist()

# # 2. Training & Predicting

# ## 2.1. Task1: How many contents do each topic have?

# ### 2.1.1. Data Preparation

# +
task1y = torch.Tensor(task1y.values)[has_content]
task1X = torch.Tensor(topics_rev.values)[has_content]
normal_features = task1X.shape[1]

task1_dataset = TensorDataset(dr_topics[has_content], task1X, task1y)
num_train = int(task1y.shape[0]*.9)
task1_train, task1_test = random_split(task1_dataset, [num_train, task1y.shape[0]-num_train])
task1_train_loader = DataLoader(task1_train, batch_size=CFG.BATCH_SIZE, shuffle=True)
task1_test_loader = DataLoader(task1_test, batch_size=CFG.BATCH_SIZE, shuffle=True)

del task1y, task1X, task1_dataset, num_train, task1_train, task1_test
gc.collect()


# -

# ### 2.1.2. Modeling

# +
# %%time

class Task1(nn.Module):
    def __init__(self, text_features, normal_features):
        super().__init__()

        self.units_text = [text_features,512,256,256,64,128,32]
        self.units_normal = [normal_features,512,256,256,64,128,32]
        self.units_concated = [self.units_text[-1]+self.units_normal[-1],128,256,256,64,128,32]
        
        self.text_l1_bn = nn.BatchNorm1d(self.units_text[0])
        self.text_l1 = nn.Linear(self.units_text[0], self.units_text[1])
        nn.init.xavier_normal_(self.text_l1.weight)
        self.text_bn = nn.ModuleList([nn.BatchNorm1d(self.units_text[i+1]) for i in range(len(self.units_text)-2)])
        self.text = nn.ModuleList([nn.Linear(self.units_text[i+1], self.units_text[i+2]) for i in range(len(self.units_text)-2)])
        
        self.normal_l1_bn = nn.BatchNorm1d(self.units_normal[0])
        self.normal_l1 = nn.Linear(self.units_normal[0], self.units_normal[1])
        nn.init.xavier_normal_(self.normal_l1.weight)
        self.normal_bn = nn.ModuleList([nn.BatchNorm1d(self.units_normal[i+1]) for i in range(len(self.units_normal)-2)])
        self.normal = nn.ModuleList([nn.Linear(self.units_normal[i+1], self.units_normal[i+2]) for i in range(len(self.units_normal)-2)])

        self.concated_l1_bn = nn.BatchNorm1d(self.units_concated[0])
        self.concated_l1 = nn.Linear(self.units_concated[0], self.units_concated[1])
        nn.init.xavier_normal_(self.concated_l1.weight)
        self.concated_bn = nn.ModuleList([nn.BatchNorm1d(self.units_concated[i+1]) for i in range(len(self.units_concated)-2)])
        self.concated = nn.ModuleList([nn.Linear(self.units_concated[i+1], self.units_concated[i+2]) for i in range(len(self.units_concated)-2)])
        self.concated_last_layer = nn.Linear(self.units_concated[-1], 1)
        self.concated_last_layer_bn = nn.BatchNorm1d(self.units_concated[-1])
        
    def forward(self, text, normal):
        """
        text features network
        """
        text = F.leaky_relu(self.text_l1(self.text_l1_bn(text)))
        for l, b in zip(self.text, self.text_bn):
            text = F.leaky_relu(l(b(text)))
        
        """
        normal features network
        """
        normal = F.leaky_relu(self.normal_l1(self.normal_l1_bn(normal)))
        for l, b in zip(self.normal, self.normal_bn):
            normal = F.leaky_relu(l(b(normal)))
            
        """
        concated network
        """
        x = torch.cat([text, normal], dim=1)
        x = F.leaky_relu(self.concated_l1(self.concated_l1_bn(x)))
        for l, b in zip(self.concated, self.concated_bn):
            x = F.leaky_relu(l(b(x)))
        
        x = F.relu(self.concated_last_layer(self.concated_last_layer_bn(x)).squeeze())
        
        return x
    
task1 = Task1(dr_topics.shape[1], normal_features)
task1.to(device)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(task1.parameters(), lr=.1)

loss_trains, loss_tests = [], []
early_stopping_count = 0

for epoch in tqdm(range(CFG.EPOCHS)):
    #"""
    #train
    #"""
    task1.train()
    loss_train = 0
    for j, (text, normal, t) in enumerate(task1_train_loader):
        text, normal, t = text.to(device), normal.to(device), t.to(device)
        y = task1(text, normal)+1
        y = torch.clamp(y, min=1, max=250)
        loss = loss_func(y, t)
        loss_train += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_train /= j+1
    loss_trains.append(loss_train)
    
    #"""
    #test
    #"""
    task1.eval()
    loss_test = 0
    preds, true_values = [], []
    for j, (text, normal, t) in enumerate(task1_test_loader):
        text, normal, t = text.to(device), normal.to(device), t.to(device)
        y = task1(text, normal)+1
        y = torch.clamp(y, min=1, max=250)
        preds += y
        true_values += t
        loss = loss_func(y, t)
        loss_test += loss.item()
    y = torch.stack(preds).squeeze()
    t = torch.stack(true_values).squeeze()
    loss_test /= j+1
    
    print(f"Epoch: {epoch+1}, TrainLoss: {loss_train**.5:.4f}, TestLoss: {loss_test**.5:.4f}.")
    print(f"y: {y.cpu().detach().numpy()}")
    
    if len(loss_tests) > 0:
        best_score_before = min(loss_tests)
    else:
        best_score_before = None
    loss_tests.append(loss_test)
    if (min(loss_tests) == loss_test) and (best_score_before != loss_test):
        early_stopping_count = 0
        torch.save(task1.state_dict(), "checkpoint/task1_bestmodel.pth")
    else:
        early_stopping_count += 1
    
    if early_stopping_count == CFG.EARLY_STOPPING_ROUNDS:
        best_epoch = epoch - CFG.EARLY_STOPPING_ROUNDS
        early_stopping_message =\
        f"Best Epoch: {best_epoch+1}" + f", TrainLoss: {loss_trains[best_epoch]**.5:.4f}"\
        + f", TestLoss: {loss_tests[best_epoch]**.5:.4f}."
        print("!!!Early Stopping !!!")
        print(early_stopping_message)
        task1.load_state_dict(torch.load("checkpoint/task1_bestmodel.pth"))
        break
        
del task1_train_loader, task1_test_loader
gc.collect()
# -

px.scatter(x = y.cpu().detach().numpy(), y = t.cpu().detach().numpy())

# ### 2.1.3. Predicting

# +
# %%time

pred_task1 = {}

for i, (topic_id, row) in enumerate(topics_only_target.iterrows()):
    if not row.has_content:
        pred_task1[topic_id] = 0
    else:
        text = dr_topics_only_target[i].to(device).view(1, -1)
        normal = torch.Tensor(row[row.index!="has_content"].values.tolist()).to(device).view(1, -1)

        task1.eval()
        y = task1(text, normal)+1
        y = torch.clamp(y, min=1, max=250).cpu().item()
        pred_task1[topic_id] = y

pred_task1

del task1
gc.collect()
# -

# ## 2.2 Task2: Which contents do each topic have?

# ### 2.2.1. Data Preparation

# +
# %%time

target_lang = list(set(target_lang) | {"en"})
positive = correlations_transform.query("t_language==c_language")
positive = positive[positive.t_language.isin(target_lang)]
positive = positive[positive.c_language.isin(target_lang)]
positive = positive[~positive.topic_id.isin(target_topics_enc)]
positive = positive[["topic_id", "content_id", "t_language"]]
positive["pair"] = 1

negative = []
for t in positive.topic_id.unique():
    res = positive[positive.topic_id==t][["content_id", "t_language"]]
    target_c_ids = lang_contentid[lang_contentid.language.isin(res.t_language)].id
    target_c_ids = set(target_c_ids) - set(res.content_id)
    c_ids = random.sample(list(target_c_ids), len(res))
    negative += [(t,c,0) for c in c_ids]
negative = pd.DataFrame(negative, columns=["topic_id", "content_id", "pair"])

data = pd.concat([positive.drop("t_language", axis=1), negative], axis=0, ignore_index=True)

X_train, X_test, y_train, y_test = tts(data[["topic_id", "content_id"]], data.pair,
                                       test_size=.1, stratify=data.pair,
                                       shuffle=True, random_state=seed)

del positive, negative, data, c_ids, target_c_ids, target_lang, lang_contentid
gc.collect()

# +
# %%time

## train data
ttext, ctext, tnormal, cnormal = [], [], [], []
for i, row in X_train.iterrows():
    ttext.append(dr_topics[dr_topics_id == row.topic_id].squeeze())
    ctext.append(dr_content[dr_content_id == row.content_id].squeeze())
    tnormal.append(torch.Tensor(topics_rev[topics_rev.index==row.topic_id].values.flatten()))
    cnormal.append(torch.Tensor(content_rev[content_rev.index==row.content_id].values.flatten()))

ttext = torch.stack(tuple(ttext))
ctext = torch.stack(tuple(ctext))
tnormal = torch.stack(tuple(tnormal))
cnormal = torch.stack(tuple(cnormal))

task2_train_dataset = TensorDataset(ttext, ctext, tnormal, cnormal, torch.Tensor(y_train.values))
task2_train_loader = DataLoader(task2_train_dataset,
                                batch_size=CFG.BATCH_SIZE,
                                shuffle=True)

print("Train Loader is created!")

## test data
ttext, ctext, tnormal, cnormal = [], [], [], []
for i, row in X_test.iterrows():
    ttext.append(dr_topics[dr_topics_id == row.topic_id].squeeze())
    ctext.append(dr_content[dr_content_id == row.content_id].squeeze())
    tnormal.append(torch.Tensor(topics_rev[topics_rev.index==row.topic_id].values.flatten()))
    cnormal.append(torch.Tensor(content_rev[content_rev.index==row.content_id].values.flatten()))

ttext = torch.stack(tuple(ttext))
ctext = torch.stack(tuple(ctext))
tnormal = torch.stack(tuple(tnormal))
cnormal = torch.stack(tuple(cnormal))

task2_test_dataset = TensorDataset(ttext, ctext, tnormal, cnormal, torch.Tensor(y_test.values))
task2_test_loader = DataLoader(task2_test_dataset,
                                batch_size=CFG.BATCH_SIZE,
                                shuffle=True)

print("Test Loader is created!")

text_normal_features = topics_rev.shape[1]

del ttext, ctext, tnormal, cnormal, topics_rev, dr_topics
gc.collect()


# -

# ### 2.2.2. Modeling

# +
# %%time

class Task2(nn.Module):
    def __init__(self, text_features, topic_features, content_features):
        super(Task2, self).__init__()
        self.units_ttext = [text_features,512,256,256,64,128,32]
        self.units_ctext = [text_features,512,256,256,64,128,32]
        self.units_tnormal = [topic_features,512,256,256,64,128,32]
        self.units_cnormal = [content_features,512,256,256,64,128,32]
        self.units_concated = [self.units_ttext[-1]+self.units_tnormal[-1],128,256,256,64,32]
        
        self.ttext_l1_bn = nn.BatchNorm1d(self.units_ttext[0])
        self.ttext_l1 = nn.Linear(self.units_ttext[0], self.units_ttext[1])
        nn.init.xavier_normal_(self.ttext_l1.weight)
        self.ttext = nn.ModuleList([nn.Linear(self.units_ttext[i+1], self.units_ttext[i+2]) for i in range(len(self.units_ttext)-2)])
        self.ttext_bn = nn.ModuleList([nn.BatchNorm1d(self.units_ttext[i+1]) for i in range(len(self.units_ttext)-2)])

        self.ctext_l1_bn = nn.BatchNorm1d(self.units_ctext[0])
        self.ctext_l1 = nn.Linear(self.units_ctext[0], self.units_ctext[1])
        nn.init.xavier_normal_(self.ctext_l1.weight)
        self.ctext = nn.ModuleList([nn.Linear(self.units_ctext[i+1], self.units_ctext[i+2]) for i in range(len(self.units_ctext)-2)])
        self.ctext_bn = nn.ModuleList([nn.BatchNorm1d(self.units_ctext[i+1]) for i in range(len(self.units_ctext)-2)])

        self.tnormal_l1_bn = nn.BatchNorm1d(self.units_tnormal[0])
        self.tnormal_l1 = nn.Linear(self.units_tnormal[0], self.units_tnormal[1])
        nn.init.xavier_normal_(self.tnormal_l1.weight)
        self.tnormal = nn.ModuleList([nn.Linear(self.units_tnormal[i+1], self.units_tnormal[i+2]) for i in range(len(self.units_tnormal)-2)])
        self.tnormal_bn = nn.ModuleList([nn.BatchNorm1d(self.units_tnormal[i+1]) for i in range(len(self.units_tnormal)-2)])

        self.cnormal_l1_bn = nn.BatchNorm1d(self.units_cnormal[0])
        self.cnormal_l1 = nn.Linear(self.units_cnormal[0], self.units_cnormal[1])
        nn.init.xavier_normal_(self.cnormal_l1.weight)
        self.cnormal = nn.ModuleList([nn.Linear(self.units_cnormal[i+1], self.units_cnormal[i+2]) for i in range(len(self.units_cnormal)-2)])
        self.cnormal_bn = nn.ModuleList([nn.BatchNorm1d(self.units_cnormal[i+1]) for i in range(len(self.units_cnormal)-2)])

        self.concated_l1_bn = nn.BatchNorm1d(self.units_concated[0])
        self.concated_l1 = nn.Linear(self.units_concated[0], self.units_concated[1])
        nn.init.xavier_normal_(self.concated_l1.weight)
        self.concated = nn.ModuleList([nn.Linear(self.units_concated[i+1], self.units_concated[i+2]) for i in range(len(self.units_concated)-2)])
        self.concated_bn = nn.ModuleList([nn.BatchNorm1d(self.units_concated[i+1]) for i in range(len(self.units_concated)-2)])
        self.concated_last_layer = nn.Linear(self.units_concated[-1], 1)
        
    def forward(self, ttext, ctext, tnormal, cnormal):
        """
        topic_text_features
        """
        ttext = F.leaky_relu(self.ttext_l1(self.ttext_l1_bn(ttext)))
        for l, b in zip(self.ttext, self.ttext_bn):
            ttext = F.leaky_relu(l(b(ttext)))
        
        """
        content_text_features
        """
        ctext = F.leaky_relu(self.ctext_l1(self.ctext_l1_bn(ctext)))
        for l, b in zip(self.ctext, self.ctext_bn):
            ctext = F.leaky_relu(l(b(ctext)))
            
        """
        topic_normal_features
        """
        tnormal = F.leaky_relu(self.tnormal_l1(self.tnormal_l1_bn(tnormal)))
        for l, b in zip(self.tnormal, self.tnormal_bn):
            tnormal = F.leaky_relu(l(b(tnormal)))

        """
        content_normal_features
        """
        cnormal = F.leaky_relu(self.cnormal_l1(self.cnormal_l1_bn(cnormal)))
        for l, b in zip(self.cnormal, self.cnormal_bn):
            cnormal = F.leaky_relu(l(b(cnormal)))

        """
        concat networks
        """
        text = torch.mul(ttext, ctext)
        normal = torch.mul(tnormal, cnormal)
        x = torch.cat([text, normal], dim=1)
        x = F.leaky_relu(self.concated_l1(self.concated_l1_bn(x)))
        for l, b in zip(self.concated, self.concated_bn):
            x = F.leaky_relu(l(b(x)))
        x = F.sigmoid(self.concated_last_layer(x)).squeeze()
        return x

    
task2 = Task2(dr_content.shape[1], text_normal_features, content_rev.shape[1])
task2.to(device);

loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(task2.parameters(), lr=.01)

loss_trains, loss_tests, accuracy_trains, accuracy_tests = [], [], [], []
early_stopping_count = 0

for epoch in tqdm(range(CFG.EPOCHS)):
    #"""
    #train
    #"""
    task2.train()
    preds = []
    true_values = []
    loss_train = 0
    for j, (ttext, ctext, tnormal, cnormal, t) in enumerate(task2_train_loader):
        ttext, ctext, tnormal, cnormal, t = ttext.to(device), ctext.to(device), tnormal.to(device), cnormal.to(device), t.to(device)
        y = task2(ttext, ctext, tnormal, cnormal)
        preds += y
        true_values += t
        loss = loss_func(y, t)
        loss_train += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_train /= j+1
    loss_trains.append(loss_train)
    y = torch.stack(preds).squeeze()
    t = torch.stack(true_values).squeeze()
    accuracy_train = (torch.sum((y > CFG.THRESHOLD)==t) / y.shape[0]).item()
    accuracy_trains.append(accuracy_train)
    
    #"""
    #test
    #"""
    task2.eval()
    preds = []
    true_values = []
    loss_test = 0
    for j, (ttext, ctext, tnormal, cnormal, t) in enumerate(task2_test_loader):
        ttext, ctext, tnormal, cnormal, t = ttext.to(device), ctext.to(device), tnormal.to(device), cnormal.to(device), t.to(device)
        y = task2(ttext, ctext, tnormal, cnormal)
        preds += y
        true_values += t
        loss = loss_func(y, t)
        loss_test += loss.item()
    loss_test /= j+1
    loss_tests.append(loss_test)
    y = torch.stack(preds).squeeze()
    t = torch.stack(true_values).squeeze()
    accuracy_test = (torch.sum((y > CFG.THRESHOLD)==t) / y.shape[0]).item()
    
    print(f"Epoch: {epoch+1}, TrainLoss: {loss_train:.4f}, TestLoss: {loss_test:.4f}, TrainAcc: {accuracy_train:.4f}, TestAcc: {accuracy_test:.4f}.")
    print(f"test y: {y.cpu().detach().numpy()}")
    
    if len(accuracy_tests) > 0:
        best_score_before = max(accuracy_tests)
    else:
        best_score_before = None
    accuracy_tests.append(accuracy_test)
    if (max(accuracy_tests) == accuracy_test) and (best_score_before != accuracy_test):
        early_stopping_count = 0
        torch.save(task2.state_dict(), "checkpoint/task2_bestmodel.pth")
    else:
        early_stopping_count += 1
    
    if early_stopping_count == CFG.EARLY_STOPPING_ROUNDS:
        best_epoch = epoch - CFG.EARLY_STOPPING_ROUNDS
        early_stopping_message =\
        f"Best Epoch: {best_epoch+1}" + f", TrainLoss: {loss_trains[best_epoch]:.4f}"\
        + f", TestLoss: {loss_tests[best_epoch]:.4f}" + f", TrainAcc: {accuracy_trains[best_epoch]:.4f}"\
        + f", TestAcc: {accuracy_tests[best_epoch]:.4f}."
        print("!!!Early Stopping !!!")
        print(early_stopping_message)
        task2.load_state_dict(torch.load("checkpoint/task2_bestmodel.pth"))
        break
# -

# ### 2.2.3. Predicting

# +
# %%time

pred_task2 = {}
task2.eval()

for i, (topic_id, row) in enumerate(topics_only_target.iterrows()):
    if pred_task1[topic_id] == 0:
        continue
        
    ttext = dr_topics_only_target[i].to(device).view(1, -1)
    tnormal = torch.Tensor(row[row.index!="has_content"].values.tolist()).to(device).view(1, -1)

    lang = row[row.index.str.startswith("language")]
    lang = lang[lang==1].index[0].split("_")[-1]
    
    target_content_or_not = content_rev[f"language_{lang}"]==1
    content_ids = target_content_or_not.index
    target_content_or_not = target_content_or_not.tolist()
    
    dataset = TensorDataset(dr_content[target_content_or_not],
                            torch.Tensor(content_rev[target_content_or_not].values))
    dataset = DataLoader(dataset, batch_size=CFG.BATCH_SIZE)
    
    preds = []
    for j, (ctext, cnormal) in enumerate(dataset):
        ctext, cnormal = ctext.to(device), cnormal.to(device)
        y = task2(ttext, ctext, tnormal, cnormal)
        preds += y 
    pred_task2[topic_id] = pd.DataFrame(
        [(k,v) for k,v in zip(content_ids, torch.stack(preds).squeeze().detach().cpu().numpy())],
        columns=["content_id", "probability"]
    ).sort_values("probability", ascending=False).head(int(pred_task1[topic_id]))
    
    print(f"finish: topic_id == {topic_id} ({lang})")
    display(pred_task2[topic_id])
# -

# # 3. Post Processing & Submitting

# +
submission = []

for k, v in pred_task2.items():
    k = t_le.inverse_transform([k])[0]
    v = c_le.inverse_transform(v.content_id)
    v = " ".join(v)
    submission.append((k,v))

submission = pd.DataFrame(submission, columns=["topic_id", "content_ids"])
submission
# -

submission.to_csv("submission.csv", index=False)


