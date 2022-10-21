import create_tfrecords
import argparse
import json
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="Customer name to identify data and saved model")
    args = parser.parse_args()
    return args
 
if __name__ == "__main__":
    args = parse_args()
    name = args.name
    #Convert input data into tfrecords
    print('Converting to tfrecords...')
    data_dir = "inputdata/"
    data_files = create_tfrecords.get_files(data_dir)
    result_path,sequence_length = create_tfrecords.create_tfrecords(data_files,name)
    #Calculate total steps in an epoch by dividing sequence length by batch size
    total_steps = int(sequence_length/16)
    warmup_steps = int(0.10 * total_steps)
    
    #Create index data file
    index_file = open("data/data.train.index","w+")
    index_file.write(result_path)
    index_file.close()
    
    config = {
      "layers": 28,
      "d_model": 4096,
      "n_heads": 16,
      "n_vocab": 50400,
      "norm": "layernorm",
      "pe": "rotary",
      "pe_rotary_dims": 64,
    
      "seq": 2048,
      "cores_per_replica": 8,
      "per_replica_batch": 1,
      "gradient_accumulation_steps": 16,
    
      "warmup_steps": warmup_steps,
      "anneal_steps": total_steps - warmup_steps,
      "lr": 5e-5,
      "end_lr": 1e-5,
      "weight_decay": 0.1,
      "total_steps": total_steps,
    
      "tpu_size": 8,
    
      "bucket": "booste-gpt-j",
      "model_dir": f"fintune_result/{name}",
    
      "train_set": "data.train.index",
      "val_set": {},
    
      "eval_harness_tasks": [
      ],
    
      "val_batches": 0,
      "val_every": total_steps + 1,
      "ckpt_every": total_steps,
      "keep_every": total_steps,
    
      "name": "",
      "wandb_project": "",
      "comment": ""
    }
    
    with open("config/model_config.json", "w") as fp:
        json.dump(config,fp,indent=4)

    os.system('python3 device_train.py --config=config/model_config.json --tune-model-path=gs://bucket/step_383500/')
    os.system('python3 slim_model.py --config=config/model_config.json')
    
