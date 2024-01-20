# End-to-end MLOps pipeline to train a Logistic regression classifier using Vertex AI on GCP

The files in this repository allow you to build a  pipeline on GCP. \
Copy the next instruction in your terminal to successfully build a pipeline using Vertex AI.

The following instructions were tested in a virtual machine with Linux OS.

 - Clone repository 
```
git clone  https://github.com/dcampanini/mlops-xgboost.git 
```
 - Create a Python virtual environment:
```
python3 -m venv env_logit
```
- Activate the virtual environment recently created
```
source env_logit/bin/activate
```
- Change to the local repository directory
```
cd mlops-logit
```
- Install the required libraries using the file requirements.txt
```
pip install -r requirements.txt
```
- Run the code to compile the pipeline
```
python pipeline_logit.py
```
- Run the code to execute the pipeline
```
python execute_pipe_logit.py
```
