# Science Agent
Language agents for data-driven scientific discovery tasks. The agent features the self-debug mechanism and shows the best performance on [ScienceAgentBench](https://github.com/OSU-NLP-Group/ScienceAgentBench). 




## Data Preparation
You can run ScienceAgent on your own data by preparing a CSV file. Each entry should have at least the following columns:
- `task_inst`: task instruction
- `domain_knowledge`: optional domain knowledge, usually annotated by subject matter experts.
- `dataset_folder_tree`: folder tree structure of the dataset(s)
- `dataset_preview`: preview of a few entries in the dataset(s)

We provide an example task from ScienceAgentBench in `benchmark/ScienceAgentBench.csv`.The additional fields are for evaluation of agent performance, which is not covered by this repository. 
For a full evaluation on ScienceAgentBench, please refer to the [ScienceAgentBench](https://github.com/OSU-NLP-Group/ScienceAgentBench) repo!

## Setup

### Environment

To start, please create a conda environment and install the necessary packages as follows:
```
conda create -n sci-agent python=3.10 pip setuptools wheel
conda activate sci-agent
pip install -r requirements.txt
```

Then, please set the `PYTHONPATH` variable to consider relative imports of local programs:
```
export PYTHONPATH=.
```

Create another environment `sci-agent-eval` for the agent to execute and debug its program via self-debug. This environment leverages `pip-tools` to update its package installations automatically for different tasks.
```
conda create -n sci-agent-eval python=3.10 pip setuptools wheel
conda activate sci-agent-eval
pip install pip-tools
conda deactivate
```
After installing `pip-tools`, please deactivate the environment. We need to keep `sci-agent-eval` clean and will only work in `sci-agent`.

### LLM Access

We currently support two kinds of LLM engines: OpenAI API and Amazon Bedrock.

To use OpenAI models (e.g. gpt-4o and o1-preview), please configure your bash shell with your OpenAI key:
```
export OPENAI_API_KEY={YOUR_OPENAI_KEY}
```

To use other LLMs (e.g. llama-3.1, mistral, and claude) on Amazon Bedrock, please setup your AWS configuration (`~/.aws/config`) as follows:
```
[default]
aws_access_key_id = {YOUR_AWS_ID}
aws_secret_access_key = {YOUR_AWS_KEY}
region=us-west-2
```

## Code Generation with Agents

### Direct Prompting and Self-Debug
You can run the agents with the following command:
```
python -u run_infer.py \
    --benchmark_name_or_path {BENCHMARK_PATH} \
    --llm_engine_name {MODEL_NAME} \
    --log_fname {LOG_FNAME} \
    [--use_knowledge] \
    [--use_self_debug] \
```
- `llm_engine_name`: name of base LLM on OpenAI or Amazon Bedrock.
- `log_fname`: your customized log file (in JSONL) name to store agent trajectories and costs, e.g. `claude_self_debug.jsonl`.
- `use_knowledge`: whether to use expert-provided knowledge or not.
- `use_self_debug`: whether to use self-debug or not (direct prompting by default).

Please see `run_science_agent.sh` for an example.

The program predicted by the agent will be saved to `pred_programs/`. It will also create two files `requirements.in` and `eval_requirements.txt`. These are intermediate files used by self-debug. Feel free to delete them after the run.

## License
Code under this repo is licensed under a MIT License.

## Acknowledgement
This work has been sponsored in part by grants from the National Science Fundation, including the ICICLE AI Institute (OAC 2112606), Amazon, and Ohio Supercomputer Center.