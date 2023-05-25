# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 1. Preparation

# +
import pandas as pd
import numpy as np
import warnings
import torch
from torch import nn
from torch.nn import functional as F
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
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
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score, confusion_matrix
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


# +
content = pd.read_csv(f'{CFG.INPUT}/content.csv')
correlations = pd.read_csv(f'{CFG.INPUT}/correlations.csv')
topics = pd.read_csv(f'{CFG.INPUT}/topics.csv')
sample_submission = pd.read_csv(f'{CFG.INPUT}/sample_submission.csv')

t_le = LabelEncoder()
topics_all = list(set(correlations.topic_id) | set(topics.id) | set(topics.parent)) + ["None"]
topics_all_enc = t_le.fit_transform(topics_all)
topics.id = t_le.transform(topics.id)
topics.parent = t_le.transform(topics.parent.fillna("None"))
correlations.topic_id = t_le.transform(correlations.topic_id)

c_le = LabelEncoder()
content.id = c_le.fit_transform(content.id)

target_topics = sample_submission.topic_id
target_topics_enc = t_le.transform(target_topics)

content.head()
correlations.head()
topics.head()
sample_submission.head()
# -

# # 2. EDA

# ## 2.1. topics

# ### 2.1.1. fundamental checking

# +
print("Numbers of topics: ", len(topics))

print("\nNull Check.")
topics.isnull().sum().reset_index()

print("\nTopic Title Duplicated.")
topics.title.duplicated().sum()

print("\nTopic Description Duplicated.")
topics.description.duplicated().sum()

for var in ["channel", "category", "level", "language", "has_content"]:
    px.bar(topics[var].value_counts(dropna=False), title=f"topic counts by {var}.")
# -

# ### 2.1.2. Network between topics

# #### network analysis

# +
# %%time

pickle_path = f"{CFG.INPUT}/G.pickle"

if os.path.exists(pickle_path):
    with open(pickle_path, 'rb') as f:
        G = pickle.load(f)
else:
    G = nx.DiGraph()
    for node in list(topics_all_enc):
      G.add_node(node)
    print("Added all nodes!")

    for i, row in topics[topics.parent.notnull()][["id", "parent"]].iterrows():
        G.add_edge(row.parent, row.id)
    print("Added all edges!")

    pos = nx.spring_layout(G, k=30, seed=1)
    for node, p in pos.items():
        G._node[node]["pos"] = p
        
    with open(pickle_path, 'wb') as f:
        pickle.dump(G, f)

# +
# %%time

with open(f"{CFG.INPUT}/G.pickle", 'wb') as p:
    pickle.dump(G, p)
# -

# ## 2.2. content

# +
print("Numbers of content: ", len(content))

print("\nNull Check.")
content.isnull().sum().reset_index()

print("\nContent Title Duplicated.")
content.title.duplicated().sum()

print("\nContent Description Duplicated.")
content.description.duplicated().sum()

for var in ["kind", "language", "copyright_holder", "license"]:
    px.bar(content[var].value_counts(dropna=False), title=f"content counts by {var}.")
# -

# ## 2.3. correlations

# +
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
        columns={col:"c_"+col for col in content.columns if col != "id"} | {"id": "content_id"}
    ),
    how="left",
    on="content_id",
)

correlations_transform = pd.merge(
    correlations_transform,
    topics[["id", "category",	"level", "language", "has_content"]].rename(
        columns={col:"t_"+col for col in topics.columns if col != "id"} | {"id": "topic_id"}
    ),
    how="left",
    on="topic_id",
)

correlations_transform

# +
for var in [col for col in correlations_transform.columns if col.startswith("t_")]:
    
    agg = correlations_transform.groupby([var, "topic_id"]).agg(
        num_contents=("content_id", "count")
    )
    px.histogram(
        data_frame = agg.reset_index(),
        x = "num_contents",
        barmode="overlay",
        histnorm='probability',
        color=var,
        title=f"histogram: numbers of content by a topic. max={agg.max().values[0]}, min={agg.min().values[0]}",
    )

    px.violin(
        data_frame = agg.reset_index(),
        x = "num_contents",
        y=var,
        color=var,
        title=f"violinplot: numbers of content by a topic. max={agg.max().values[0]}, min={agg.min().values[0]}",
    )

agg = correlations_transform.groupby(["topic_id"]).agg(
    num_contents=("content_id", "count")
)
px.histogram(
    data_frame = agg.reset_index(),
    x = "num_contents",
    barmode="overlay",
    histnorm='probability',
    title=f"histogram: numbers of content by a topic. max={agg.max().values[0]}, min={agg.min().values[0]}",
)

# +
for var in [col for col in correlations_transform.columns if col.startswith("c_") and col != "c_copyright_holder"]:
    
    agg = correlations_transform.groupby([var, "content_id"]).agg(
        num_topics=("topic_id", "count")
    )
    px.histogram(
        data_frame = agg.reset_index(),
        x = "num_topics",
        barmode="overlay",
        histnorm='probability',
        color=var,
        title=f"histogram: number of topics by a content. max={agg.max().values[0]}, min={agg.min().values[0]}",
    )
    px.box(
        data_frame = agg.reset_index(),
        y = "num_topics",
        x = var,
        color=var,
        title=f"boxplot: number of topics by a content. max={agg.max().values[0]}, min={agg.min().values[0]}",
    )

agg = correlations_transform.groupby(["content_id"]).agg(
    num_topics=("topic_id", "count")
)
px.histogram(
    data_frame = agg.reset_index(),
    x = "num_topics",
    barmode="overlay",
    histnorm='probability',
    title=f"histogram: number of topics by a content. max={agg.max().values[0]}, min={agg.min().values[0]}",
)
# -

agg = pd.crosstab(correlations_transform.t_language, columns=correlations_transform.c_language)
agg.shape
agg

# ## memo
# + Task1: How many topics do each content belong to?(from 1 to 241)
# + Task2: How many contents do each topic have? (from 1 to 293)
# + Task3: Which contents do each topic have?

# +
text_cols = ["title", "description", "text"]
agg = correlations_transform.groupby(["content_id"]).agg(
    num_topics=("topic_id", "count")
).reset_index().rename(columns={"content_id":"id"})
content_rev = pd.merge(content, agg, how="left", on="id").drop(text_cols, axis=1)
content_rev.num_topics = content_rev.num_topics.fillna(0)
content_rev["text"] = content.title.fillna("") + \
    "\n" + content.description.fillna("") + \
    "\n" + content.text.fillna("")
content_rev

text_cols = ["title", "description"]
agg = correlations_transform.groupby(["topic_id"]).agg(
    num_contents=("content_id", "count")
).fillna(0).reset_index().rename(columns={"topic_id":"id"})
topics_rev = pd.merge(topics, agg, how="left", on="id").drop(text_cols, axis=1)
topics_rev.num_contents = topics_rev.num_contents.fillna(0)
topics_rev["text"] = topics.title.fillna("") + "\n" + topics.description.fillna("")

topics_rev
# -

# # 3. Training

# ## 3.0.Feature Engineering

# ### 3.0.1. Text Embedding

# +
# %%time
gc.collect()

model = AutoModel.from_pretrained(CFG.MODEL)
tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL)
model.eval();
model.to(device);

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

if not os.path.exists(f"{CFG.INPUT}/dr_topics.pickle"):
    # %time dr_topics = distributed_representations(topics_rev.text) # Wall time: 7min 27s
    with open(f"{CFG.INPUT}/dr_topics.pickle", 'wb') as p:
        pickle.dump(dr_topics, p)
else:
    with open(f"{CFG.INPUT}/dr_topics.pickle", 'rb') as f:
        dr_topics = pickle.load(f)

if not os.path.exists(f"{CFG.INPUT}/dr_content.pickle"):
    # %time dr_content = distributed_representations(content_rev.text) # Wall time: 44min 49s
    with open(f"{CFG.INPUT}/dr_content.pickle", 'wb') as p:
        pickle.dump(dr_content, p)
else:
    with open(f"{CFG.INPUT}/dr_content.pickle", 'rb') as f:
        dr_content = pickle.load(f)
# -

# ### 3.0.2. Analyze Channels

channel = topics_rev.groupby("channel").agg(ch_num_topics=("id", "count"))
channel = channel.join(topics_rev.groupby("channel").agg(ch_variety_lang=("language", "nunique")))
channel = channel.join(topics_rev.groupby("channel").agg(ch_max_level=("level", "max")))
channel = channel.join(topics_rev.groupby("channel").agg(ch_mean_level=("level", "mean")))
channel = channel.join(topics_rev.groupby("channel").agg(ch_std_level=("level", "std")))
channel = channel.reset_index()
channel

# ### 3.0.3. Contents: Copyright & Text Processing

# +
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
content_rev["words_text"] = [len(t) for t in content_rev.text.str.split(" ")]

content_rev
# -

# ### 3.0.4. Topics: Text Processing

topics_rev["words_text"] = [len(t) for t in topics_rev.text.str.split(" ")]
topics_rev

# ## 3.1. Task1: How many topics do each content belong to?

# ### 3.1.1. Data Preparation

# +
task1y = content_rev.num_topics.copy()
task1X = content_rev.drop(["text", "num_topics"], axis=1).set_index("id").fillna(0)
task1X = pd.get_dummies(task1X, columns=["kind", "language", "copyright_holder", "license"], dummy_na=True)

task1y = torch.Tensor(task1y.values)
task1X = torch.Tensor(task1X.values)
task1X_backup = task1X.clone()
task1X = torch.cat([task1X, dr_content], dim=1)

all_idx = range(len(task1y))
train_idx = random.sample(all_idx, int(len(task1y)*0.8))
test_idx = list(set(all_idx) - set(train_idx))

task1X
# -

# ### 3.2.1. Modeling

# #### 3.2.1.1. LightGBM

# +
train_lgb = lgb.Dataset(task1X[train_idx].numpy(), task1y[train_idx].numpy())
eval_lgb = lgb.Dataset(task1X[test_idx].numpy(), task1y[test_idx].numpy())

params_lgb = dict(
            task="train",
            boosting_type="gbdt",
            objective="regression",
            metric="rmse",
            learning_rate=.05,
            force_col_wise=True,
            device='cpu',
            num_leaves=700,
            max_iter=2000,
            seed=seed,
            lambda_l1=.1,
            lambda_l2=.1,
            bagging_freq=10,
            bagging_seed=seed,
        )

evals_result = {}

model_lgb = lgb.train(
    params=params_lgb,
    train_set=train_lgb,
    valid_names=['train', 'valid'],
    valid_sets=[train_lgb, eval_lgb],
    early_stopping_rounds=20,
    evals_result=evals_result,
    #verbose_eval=self.verbose
)

preds_lgb = model_lgb.predict(task1X[test_idx].numpy())
res = mse(preds_lgb, task1y[test_idx].numpy())**.5
res
# -

px.scatter(y=preds_lgb, x=task1y[test_idx].numpy())

# ## 3.2. Task2: How many contents do each topic have?

# ### 3.2.1. Data Preparation

# +
task2y = topics_rev[topics_rev.has_content].num_contents.copy()
task2X = pd.merge(topics_rev, channel, how="left", on="channel")
task2X = task2X.drop(["channel", "num_contents", "text", "parent", "has_content"], axis=1).set_index("id")
task2X = pd.get_dummies(task2X, columns=["category", "language"], dummy_na=True).fillna(0)
task2X_backup = task2X.copy()

task2y = torch.Tensor(task2y.values)
task2X = torch.Tensor(task2X.values)[topics_rev.has_content]
#task2X = torch.cat([task2X, dr_topics[topics_rev.has_content]], dim=1)

task2_dataset = TensorDataset(dr_topics[topics_rev.has_content], task2X, task2y)
num_train = int(task2y.shape[0]*.9)
task2_train, task2_test = random_split(task2_dataset, [num_train, task2y.shape[0]-num_train])
task2_train_loader = DataLoader(task2_train, batch_size=CFG.BATCH_SIZE, shuffle=True)
task2_test_loader = DataLoader(task2_test, batch_size=CFG.BATCH_SIZE, shuffle=True)


# -

# ### 3.2.2. Modeling

# #### 3.2.2.1. Pytorch

# +
# %%time

class Task2(nn.Module):
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
    
task2 = Task2(dr_topics.shape[1], task2X.shape[1])
task2.cuda()

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(task2.parameters(), lr=.1)

loss_trains, loss_tests = [], []
early_stopping_count = 0

for epoch in tqdm(range(CFG.EPOCHS)):
    #"""
    #train
    #"""
    task2.train()
    loss_train = 0
    for j, (text, normal, t) in enumerate(task2_train_loader):
        text, normal, t = text.cuda(), normal.cuda(), t.cuda()
        y = task2(text, normal)+1
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
    task2.eval()
    loss_test = 0
    preds, true_values = [], []
    for j, (text, normal, t) in enumerate(task2_test_loader):
        text, normal, t = text.cuda(), normal.cuda(), t.cuda()
        y = task2(text, normal)+1
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
        torch.save(task2.state_dict(), "checkpoint/task2_bestmodel.pth")
    else:
        early_stopping_count += 1
    
    if early_stopping_count == CFG.EARLY_STOPPING_ROUNDS:
        best_epoch = epoch - CFG.EARLY_STOPPING_ROUNDS
        early_stopping_message =\
        f"Best Epoch: {best_epoch+1}" + f", TrainLoss: {loss_trains[best_epoch]**.5:.4f}"\
        + f", TestLoss: {loss_tests[best_epoch]**.5:.4f}."
        print("!!!Early Stopping !!!")
        print(early_stopping_message)
        task2.load_state_dict(torch.load("checkpoint/task2_bestmodel.pth"))
        break
# -

px.scatter(x = y.cpu().detach().numpy(), y = t.cpu().detach().numpy())

# #### 3.2.2.2. LightGBM

# +
all_idx = range(len(task2y))
train_idx = random.sample(all_idx, int(len(task2y)*0.8))
test_idx = list(set(all_idx) - set(train_idx))

train_lgb = lgb.Dataset(task2X[train_idx].numpy(), task2y[train_idx].numpy())
eval_lgb = lgb.Dataset(task2X[test_idx].numpy(), task2y[test_idx].numpy())

# +
params_lgb = dict(
            task="train",
            boosting_type="gbdt",
            objective="regression",
            metric="rmse",
            learning_rate=.05,
            force_col_wise=True,
            device='cpu',
            num_leaves=700,
            max_iter=2000,
            seed=seed,
            lambda_l1=.1,
            lambda_l2=.1,
            bagging_freq=10,
            bagging_seed=seed,
        )

evals_result = {}

model_lgb = lgb.train(
    params=params_lgb,
    train_set=train_lgb,
    valid_names=['train', 'valid'],
    valid_sets=[train_lgb, eval_lgb],
    early_stopping_rounds=20,
    evals_result=evals_result,
    #verbose_eval=self.verbose
)

preds_lgb = model_lgb.predict(task2X[test_idx].numpy())
res = mse(preds_lgb, task2y[test_idx].numpy())**.5
res
# -

px.scatter(y=preds_lgb, x=task2y[test_idx].numpy())

# ## 3.3 Task3: Which contents do each topic have?

# ### 3.3.1. Data Preparation

# +
# %%time

target_lang = topics_rev[topics_rev.id.isin(target_topics_enc)].language
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
    target_c_ids = content_rev[content_rev.language.isin(res.t_language)].id
    target_c_ids = set(target_c_ids) - set(res.content_id)
    c_ids = random.sample(list(target_c_ids), len(res))
    negative += [(t,c,0) for c in c_ids]
negative = pd.DataFrame(negative, columns=["topic_id", "content_id", "pair"])

data = pd.concat([positive.drop("t_language", axis=1), negative], axis=0, ignore_index=True)

X_train, X_test, y_train, y_test = tts(data[["topic_id", "content_id"]], data.pair,
                                       test_size=.1, stratify=data.pair,
                                       shuffle=True, random_state=seed)

del positive, negative, data, c_ids, target_c_ids, target_lang
gc.collect()

# +
# %%time

## train data
ttext, ctext, tnormal, cnormal = [], [], [], []
for i, row in X_train.iterrows():
    ttext.append(dr_topics[row.topic_id])
    ctext.append(dr_content[row.content_id])
    tnormal.append(torch.Tensor(task2X_backup[task2X_backup.index==row.topic_id].values.flatten()))
    cnormal.append(task1X_backup[row.content_id])

ttext = torch.stack(tuple(ttext))
ctext = torch.stack(tuple(ctext))
tnormal = torch.stack(tuple(tnormal))
cnormal = torch.stack(tuple(cnormal))

task3_train_dataset = TensorDataset(ttext, ctext, tnormal, cnormal, torch.Tensor(y_train.values))
task3_train_loader = DataLoader(task3_train_dataset,
                                batch_size=CFG.BATCH_SIZE,
                                shuffle=True)

## test data
ttext, ctext, tnormal, cnormal = [], [], [], []
for i, row in X_test.iterrows():
    ttext.append(dr_topics[row.topic_id])
    ctext.append(dr_content[row.content_id])
    tnormal.append(torch.Tensor(task2X_backup[task2X_backup.index==row.topic_id].values.flatten()))
    cnormal.append(task1X_backup[row.content_id])

ttext = torch.stack(tuple(ttext))
ctext = torch.stack(tuple(ctext))
tnormal = torch.stack(tuple(tnormal))
cnormal = torch.stack(tuple(cnormal))

task3_test_dataset = TensorDataset(ttext, ctext, tnormal, cnormal, torch.Tensor(y_test.values))
task3_test_loader = DataLoader(task3_test_dataset,
                                batch_size=CFG.BATCH_SIZE,
                                shuffle=True)


# -

# ### 3.3.2. Modeling

# + jupyter={"outputs_hidden": true}
# %%time

class Task3(nn.Module):
    def __init__(self, text_features, topic_features, content_features):
        super(Task3, self).__init__()
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

    
task3 = Task3(dr_topics.shape[1], task2X_backup.shape[1], task1X_backup.shape[1])
task3.cuda();

loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(task3.parameters(), lr=.01)

loss_trains, loss_tests, accuracy_trains, accuracy_tests = [], [], [], []
early_stopping_count = 0

for epoch in tqdm(range(CFG.EPOCHS)):
    #"""
    #train
    #"""
    task3.train()
    preds = []
    loss_train = 0
    for j, (ttext, ctext, tnormal, cnormal, t) in enumerate(task3_train_loader):
        ttext, ctext, tnormal, cnormal, t = ttext.cuda(), ctext.cuda(), tnormal.cuda(), cnormal.cuda(), t.cuda()
        y = task3(ttext, ctext, tnormal, cnormal)
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
    task3.eval()
    preds = []
    true_values = []
    loss_test = 0
    for j, (ttext, ctext, tnormal, cnormal, t) in enumerate(task3_test_loader):
        ttext, ctext, tnormal, cnormal, t = ttext.cuda(), ctext.cuda(), tnormal.cuda(), cnormal.cuda(), t.cuda()
        y = task3(ttext, ctext, tnormal, cnormal)
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
        torch.save(task3.state_dict(), "checkpoint/task3_bestmodel.pth")
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
        task3.load_state_dict(torch.load("checkpoint/task3_bestmodel.pth"))
        break


# +
# %%time

class Task3(nn.Module):
    def __init__(self, text_features, topic_features, content_features):
        super(Task3, self).__init__()
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

    
task3 = Task3(dr_topics.shape[1], task2X_backup.shape[1], task1X_backup.shape[1])
task3.cuda();

loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(task3.parameters(), lr=.01)

loss_trains, loss_tests, accuracy_trains, accuracy_tests = [], [], [], []
early_stopping_count = 0

for epoch in tqdm(range(CFG.EPOCHS)):
    #"""
    #train
    #"""
    task3.train()
    preds = []
    true_values = []
    loss_train = 0
    for j, (ttext, ctext, tnormal, cnormal, t) in enumerate(task3_train_loader):
        ttext, ctext, tnormal, cnormal, t = ttext.cuda(), ctext.cuda(), tnormal.cuda(), cnormal.cuda(), t.cuda()
        y = task3(ttext, ctext, tnormal, cnormal)
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
    task3.eval()
    preds = []
    true_values = []
    loss_test = 0
    for j, (ttext, ctext, tnormal, cnormal, t) in enumerate(task3_test_loader):
        ttext, ctext, tnormal, cnormal, t = ttext.cuda(), ctext.cuda(), tnormal.cuda(), cnormal.cuda(), t.cuda()
        y = task3(ttext, ctext, tnormal, cnormal)
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
        torch.save(task3.state_dict(), "checkpoint/task3_bestmodel.pth")
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
        task3.load_state_dict(torch.load("checkpoint/task3_bestmodel.pth"))
        break
# -

target_topics


