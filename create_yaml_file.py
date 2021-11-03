import copy
import subprocess
import yaml

with open("yaml_files/default_experiment.yaml", 'r') as f:
    default_yaml = f.read()

cluster = "ai2/on-prem-mosaic"
tasks = ['SST-2']
prompt_tasks = ['subtask047_misc_answering_science_questions']
gammas = [0.01]

d1 = yaml.load(default_yaml)

for gamma in gammas:
    for task in tasks:
        for prompt_task in prompt_tasks:
            d = copy.deepcopy(d1)

            d['tasks'][0]['context']['cluster'] = cluster

            name = f"experiment_task={task}-prompt_task={prompt_task}-gamma={gamma}"
            d['description'] = name
            task_idx = 6
            assert d['tasks'][0]['arguments'][task_idx] == 'SST-2'
            d['tasks'][0]['arguments'][task_idx] = task

            prompt_task_idx = 8
            assert d['tasks'][0]['arguments'][prompt_task_idx] == 'subtask047_misc_answering_science_questions'
            d['tasks'][0]['arguments'][prompt_task_idx] = prompt_task

            gamma_idx = 28
            assert d['tasks'][0]['arguments'][gamma_idx] == 0.001
            d['tasks'][0]['arguments'][gamma_idx] = gamma

            print(d)

            fn = "yaml_files/{}.yaml".format(name)
            file = open(fn, "w")
            yaml.dump(d, file, default_flow_style=True)
            file.close()

            cmd = "beaker experiment create {} --workspace danielk/prompting".format(fn)
            subprocess.Popen(cmd, shell=True)

