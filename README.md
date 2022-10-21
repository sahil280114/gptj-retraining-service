# gptj-retraining-service

### To retrain GPT-J on a dataset
 * SSH into TPU VM
 * Pull the latest repo
 * Create a gcp service account with admin access to google cloud storage and save the service account json file as "service_account.json"
 * `pip install -r requirements.txt`
 * Put input data into the 'inputdata\' dir (as a .txt file, or multiple .txt files)
 * in app.py line 52-53 add a gcs bucket name and dir name to define where to store finetune results
 * in app.py line 74, specify a gcs location where full gpt-j weights are stored
 * Run- `python3 app.py {name}` (name to identify the data and finetune version)

The fintuned are stored in GCS Bucket specified in app.py
