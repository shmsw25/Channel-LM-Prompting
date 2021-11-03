# Steps to running Beaker
###Need to be done once only: 
- Step 1: install beaker on your machine: https://beaker.org/user
- Step 1.1: create a workspace: https://beaker.org/workspaces


### Few times, only when you change the data:
- Step 2: build beaker dataset for your data: `beaker dataset create --name prompting_project_datasets ./data --workspace [WORKSPACE]`
 
- Step 3: build docker image for the repo: `docker build -t prompting_project .`
- Step 3.1: build beaker image for our docker image: `beaker image create -n prompting_project prompting_project --workspace [WORKSPACE]`


### the main items which is typically quick
- Step 4: set the parameters in the .yaml file (dataset name(s), image names (s), cluster name, parameters of your experiment, etc)
- Step 4.1: run the experiment on beaker: `beaker experiment create -f experiment.yaml --workspace [WORKSPACE]`
