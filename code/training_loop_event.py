import pandas as pd
import numpy as np  
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import json 
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix, f1_score, precision_score, recall_score, mean_absolute_error
from datasets import Dataset
from transformers import RobertaTokenizer, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import torch
import torch.nn.functional as F
from torch import nn
from transformers import RobertaModel, RobertaConfig
from transformers.modeling_outputs import SequenceClassifierOutput

# Loading data 
df_train = pd.read_csv('train_data.csv')
df_test = pd.read_csv('test_data.csv')

# Handling missing values to avoid nan type errors 
df_train['entity_candidate'] = df_train['entity_candidate'].fillna("")
df_train['event_candidate'] = df_train['event_candidate'].fillna("")
df_test['entity_candidate'] = df_test['entity_candidate'].fillna("")
df_test['event_candidate'] = df_test['event_candidate'].fillna("")

# Designing model inputs 
# not providing probabilities to test set to check 
# model's ability to predict them from learned examples.
df_train['input'] = df_train.apply(
    lambda row: (
        f"Example: {row['text']} "
        f"[PRONOUN: {row['pronoun']}] "
        f"[ENTITY: {row['entity_candidate']}] "
        f"[EVENT: {row['event_candidate']}] "
        f"[ENTITY_PROBABILITY: {row['entity_prob']}] "
        f"[EVENT_PROBABILITY: {row['event_prob']}]"
    ), axis=1
)

# For the test DataFrame (which doesn’t include probability tokens):
df_test['input'] = df_test.apply(
    lambda row: (
        f"Example: {row['text']} "
        f"[PRONOUN: {row['pronoun']}] "
        f"[ENTITY: {row['entity_candidate']}] "
        f"[EVENT: {row['event_candidate']}]"
    ), axis=1
)



# Mapping entity and event probabilities into
# one columns called labels

df_train['labels'] = df_train['event_prob'].astype(float)
df_test['labels'] = df_test['event_prob'].astype(float)

# Coverting to dataset class 
train_dataset = Dataset.from_pandas(df_train[["input", "labels"]])
test_dataset = Dataset.from_pandas(df_test[["input", "labels"]])

# Shuffling the train dataset
train_dataset = train_dataset.shuffle(seed=42)


# Loading model and tokenizer 
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Helper function to tokenize inputs 
def tokenize_function(examples):
    return tokenizer(examples["input"],  padding="max_length", truncation=True, max_length=512)


# Tokenizing the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Removing origian raw inputs from the datasets 
# to save space 
train_dataset = train_dataset.remove_columns(["input"])
test_dataset = test_dataset.remove_columns(["input"])

# Customizing roberta model for predicting probabilities 

class RobertaForProbabilities(nn.Module):
    def __init__(self, model_name="roberta-base", drop_out_rate=0.3):
        super(RobertaForProbabilities, self).__init__()
        self.config = RobertaConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = drop_out_rate 
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.probability_head = nn.Linear(self.config.hidden_size, 1)  # One output for entity 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, input_ids, attention_mask, labels=None):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if labels is not None:
            labels = labels.to(self.device)
            
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]


        pooled_output = self.dropout(pooled_output)
        event_prob_logit = self.probability_head(pooled_output).squeeze(-1)
        #entity_probs = torch.sigmoid(entity_prob_logit)
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            #loss = loss_fct(entity_probs, labels.view(-1).float())
            loss = loss_fct(event_prob_logit, labels.view(-1).float())


        return SequenceClassifierOutput(
            loss=loss,
            #logits=entity_probs,
            logits=event_prob_logit,  
      
    )


    
# Loading the model (object of customized roberta class) 
model = RobertaForProbabilities(model_name="roberta-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Setting up training arguments
training_args = TrainingArguments(
    output_dir="./roberta_prob_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="linear",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    max_grad_norm=1.0,
    logging_dir='./logs'
)

# Defining a custom compute metrics function 
def compute_metrics(eval_pred): 
    preds, labels = eval_pred
    preds = torch.tensor(preds).float().view(-1)
    labels = torch.tensor(labels).float().view(-1)
    preds = torch.sigmoid(preds)
    mse = F.mse_loss(torch.tensor(preds), torch.tensor(labels)).item()
    mae = F.l1_loss(preds, labels).item()
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
    }
    
# Defining the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics, 
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Training the model
trainer.train()

# Evaluating the model
eval_results = trainer.evaluate(eval_dataset=test_dataset)


print(f"MSE: {eval_results['eval_mse']:.4f}")
print(f"RMSE: {eval_results['eval_rmse']:.4f}")
print(f"MAE: {eval_results['eval_mae']:.4f}")




# Saving the model
model_save_path = "./roberta_prob_model"
os.makedirs(model_save_path, exist_ok=True)
torch.save(model.state_dict(), os.path.join(model_save_path, "pytorch_model.bin"))
model.config.save_pretrained(model_save_path)


# Getting the predictions
preds = trainer.predict(test_dataset)

# Getting entity reference predictions 
# getting event reference predictions as
# compliment of entity reference predictions. 
pred_logits = preds.predictions
event_pred_probs = torch.from_numpy(preds.predictions).float()
entity_pred_probs = 1 - event_pred_probs 

labels = []

for event_prob in event_pred_probs:
    event_item = round(event_prob.item(), 2)
    entity_item = 1 - round(1 - event_item, 2)
    
    if event_item == 0: 
        labels.append("Entity") 
        
    elif entity_item == 0:
        labels.append("Event")
        
    elif 0.61 <= event_item < 0.9 or 0.1 <= entity_item < 0.39:
        labels.append("Event Leaning")
        
    elif 0.61 <= entity_item < 0.9 or 0.1 <= event_item < 0.39:
        labels.append("Entity Leaning")
        
    elif 0.4 <= entity_item < 0.6 and 0.4 <= event_item < 0.6:
        labels.append("Ambiguous")
        
    else:
        labels.append("Unknown")

results_df = pd.DataFrame({
    "text": df_test['text'],
    "pronoun": df_test["pronoun"],
    "entity_candidate": df_test["entity_candidate"],
    "event_candidate": df_test["event_candidate"],
    "entity_prob": entity_pred_probs.numpy(),
    "event_prob": event_pred_probs.numpy(),
    "label": labels
})

# Save to CSV
results_df.to_csv("predicted_event_prob_labels_with_thresholding.csv", index=False)
print("Results with threshold labels saved to predicted_event_prob_labels_with_thresholding.csv")



# baseline comparison 


true_y = df_test['labels'].values
pred_y = event_pred_probs.numpy() 

# Mean baseline 
mean_baseline = df_train['labels'].mean()
y_pred_mean = np.full_like(true_y, fill_value=mean_baseline)

# Median baseline 
median_baseline = df_train['labels'].median()
y_pred_median = np.full_like(true_y, fill_value=median_baseline)

# Random values baseline
np.random.seed(42)  # For reproducibility
y_pred_random = np.random.rand(len(true_y))

# Calculating metrics for each baseline

def evaluation_metrics(pred_y, label="Model"):
    mse = mean_squared_error(true_y, pred_y) 
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_y, pred_y)
    print(f"{label} → MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

print("My model evaluation metrics:", evaluation_metrics(pred_y, label="My Model"))
print("Mean baseline evaluation metrics:", evaluation_metrics(y_pred_mean, label="Mean Baseline"))
print("Median baseline evaluation metrics:", evaluation_metrics(y_pred_median, label="Median Baseline"))
print("Random baseline evaluation metrics:", evaluation_metrics(y_pred_random, label="Random Baseline"))

print("Initial few examples Predicted probabilities predictions:")
print("Entity probabilities: ", entity_pred_probs[:5])
print("Event probabilities: ", event_pred_probs[:5])
output_file = "prob_predictions_with_event_prob.txt"
# Storing predictions in a text file
with open(output_file, "w", encoding="utf-8") as f:
    for i, prob in enumerate(event_pred_probs):
        entry = {
            "example_index": i,
            "entity_probability": (1 - prob).item(),
            "event_probability": prob.item()
        } 
        f.write(json.dumps(entry) + "\n")



# ploting the loss curve 
log_hist = trainer.state.log_history
train_epochs = []
train_losses = []
val_epochs = []
val_losses = []

for entry in log_hist:
    if 'loss' in entry and 'epoch' in entry:
        train_epochs.append(entry['epoch'])
        train_losses.append(entry['loss'])
    if 'eval_loss' in entry and 'epoch' in entry:
        val_epochs.append(entry['epoch'])
        val_losses.append(entry['eval_loss'])

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(train_epochs, train_losses, label='Training Loss', marker='o')
plt.plot(val_epochs, val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve for Event Probability Prediction')
plt.legend()
plt.grid(True)
plt.savefig("roberta_loss_curve_event_prob.png")
plt.show()


# Note: AI Assistants were used to debug and refine the code.