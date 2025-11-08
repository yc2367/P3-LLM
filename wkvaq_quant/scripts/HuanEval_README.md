To evaluate Llama-3.1-8B-Instruct and beyond, you must install "lm-eval==0.4.9".
    pip install --no-deps --force-reinstall -v "lm-eval==0.4.9"

Go to the installed "lm-eval" package in the virtual conda environment.
    E.g.: /home/yc2367/anaconda3/envs/p2-llm/lib/python3.10/site-packages/lm_eval/tasks/humaneval/

Go to "utils.py" under the "humaneval/" folder. 
Change this line of code
```
doc["prompt"] + (r if r.rfind("```") == -1 else r[: r.rfind("```")])
```
to 
```
doc["prompt"] + (r if r.find("```") == -1 else r[: r.find("```")])
```

Go to "humaneval_instruct.yaml" under the "humaneval/" folder. 
Change this line of code
```
doc_to_text: "Write a solution to the following problem and make sure that it passes the tests:\n```{{prompt}}"
``` 
to 
```
doc_to_text: "Write a solution to the following problem and make sure that it passes the tests:\n```{{prompt}}```"
``` 