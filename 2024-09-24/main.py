import os

model_dir = os.environ['SM_MODEL_DIR']

with open(model_dir + '/output.txt', 'w') as f:
    f.write('Ciao')
    
output_dir = os.environ['SM_OUTPUT_DIR']
with open(output_dir + '/output.txt', 'w') as f:
    f.write('Ciao')