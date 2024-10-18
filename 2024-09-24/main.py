import os

model_dir = os.environ['SM_MODEL_DIR']

with open(model_dir + '/output_model.txt', 'w') as f:
    f.write('Ciao, sono il modello di ML!')
    
output_dir = os.environ['SM_OUTPUT_DIR']
with open(output_dir + '/output_data.txt', 'w') as f:
    f.write('Ciao, sono i LOG del training!')
    
input_dir = os.environ['SM_INPUT_DIR']    

lines = 0
with open(input_dir + "/data/training/train.csv", 'r') as f:
    lines = len(f.readlines())
    print("#######Total number of lines:", lines)
          
with open(output_dir + "/output_lines.txt", 'w') as f:
    f.write(f"Ciao, le righe sono {lines}")