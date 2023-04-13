#!/usr/bin/env python3
import faiss
import os
from collections import deque
from typing import Dict, List, Optional, Any

from langchain import LLMChain, OpenAI, PromptTemplate, SerpAPIWrapper
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.chains.base import Chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.docstore import InMemoryDocstore
from pydantic import BaseModel, Field

import importlib
import time
import subprocess

import openai
import pinecone
from dotenv import load_dotenv

# Load default environment variables (.env)
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")
assert OPENAI_API_MODEL, "OPENAI_API_MODEL environment variable is missing from .env"

if "gpt-4" in OPENAI_API_MODEL.lower():
    print(
        "\033[91m\033[1m"
        + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + "\033[0m\033[0m"
    )

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
assert PINECONE_API_KEY, "PINECONE_API_KEY environment variable is missing from .env"

PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
assert (
    PINECONE_ENVIRONMENT
), "PINECONE_ENVIRONMENT environment variable is missing from .env"

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
assert SERPAPI_API_KEY, "SERPAPI_API_KEY environment variable is missing from .env"

# Table config
YOUR_TABLE_NAME = os.getenv("TABLE_NAME", "")
assert YOUR_TABLE_NAME, "TABLE_NAME environment variable is missing from .env"

# Goal configuation
OBJECTIVE = os.getenv("OBJECTIVE", "")
INITIAL_TASK = os.getenv("INITIAL_TASK", os.getenv("FIRST_TASK", ""))

# Extensions support begin


def can_import(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


DOTENV_EXTENSIONS = os.getenv("DOTENV_EXTENSIONS", "").split(" ")

# Command line arguments extension
# Can override any of the above environment variables
ENABLE_COMMAND_LINE_ARGS = (
    os.getenv("ENABLE_COMMAND_LINE_ARGS", "false").lower() == "true"
)
if ENABLE_COMMAND_LINE_ARGS:
    if can_import("extensions.argparseext"):
        from extensions.argparseext import parse_arguments

        OBJECTIVE, INITIAL_TASK, OPENAI_API_MODEL, DOTENV_EXTENSIONS = parse_arguments()

# Load additional environment variables for enabled extensions
if DOTENV_EXTENSIONS:
    if can_import("extensions.dotenvext"):
        from extensions.dotenvext import load_dotenv_extensions

        load_dotenv_extensions(DOTENV_EXTENSIONS)

# TODO: There's still work to be done here to enable people to get
# defaults from dotenv extensions # but also provide command line
# arguments to override them

# Extensions support end

# Check if we know what we are doing
assert OBJECTIVE, "OBJECTIVE environment variable is missing from .env"
assert INITIAL_TASK, "INITIAL_TASK environment variable is missing from .env"

if "gpt-4" in OPENAI_API_MODEL.lower():
    print(
        "\033[91m\033[1m"
        + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + "\033[0m\033[0m"
    )

# Print OBJECTIVE
print("\033[94m\033[1m" + "\n*****OBJECTIVE*****\n" + "\033[0m\033[0m")
print(f"{OBJECTIVE}")

print("\033[93m\033[1m" + "\nInitial task:" +
      "\033[0m\033[0m" + f" {INITIAL_TASK}")


# TaskManagementChain is a unification of the formerly separate agents
# (TaskCreationChain and TaskPrioritizationChain)
class TaskManagementChain(LLMChain):
    """Chain to manage tasks, including creation and prioritization."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_management_template = (
            "You are a COO and Head of Product that is fully responsible for achieving the ultimate (important & urgent) objective of your team: {objective}. You do this by owning and managing the 'backlog' of specific, concrete tasks to be done by your small team of workers, considering the context, i.e. progress toward the objective, and recent task & result."
            " - Last completed task: '{task_description}'. It had this result: '{result}'."
            " - Current backlog (list) of incomplete tasks: {incomplete_tasks}."
            " Based on this, you create a refreshed backlog - adding and/or removing tasks, and prioritizing (ranking) them all as you deem fit. Tasks must be very bite-sized, outcome-oriented, and described clearly, with detailed suggestions & success criteria. You consistently share the updated priorities with the team as a numbered list, e.g.:"
            " 1. Search for 'New AI products & startups, March 2023'."
            " 2. Write a highly opinionated, insightful, engaging, excellent draft of a blog article on this topic, based on the research findings."
            " You always start the task list with the number {next_task_id}."
        )
        prompt = PromptTemplate(
            template=task_management_template,
            input_variables=["result", "task_description",
                             "incomplete_tasks", "next_task_id", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class BabyAGI(Chain, BaseModel):
    """Controller model for the BabyAGI agent."""

    task_list: deque = Field(default_factory=deque)
    task_management_chain: TaskManagementChain = Field(...)
    execution_chain: AgentExecutor = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = None

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

    def print_next_task(self, task: Dict):
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

    def print_task_result(self, result: str):
        print("\033[93m\033[1m" +
              "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return []

    def create_and_prioritize_tasks(self, task_management_chain: LLMChain, this_task_id: int, task_list: List[Dict], result: Dict, task_description: str, objective: str) -> List[Dict]:
        """Create new tasks and prioritize tasks."""
        incomplete_tasks = ", ".join([t["task_name"] for t in task_list])
        response = task_management_chain.run(
            result=result, task_description=task_description, incomplete_tasks=incomplete_tasks, next_task_id=this_task_id+1, objective=objective)

        new_tasks = response.split('\n')
        prioritized_task_list = []
        for task_string in new_tasks:
            if not task_string.strip():
                continue
            task_parts = task_string.strip().split(".", 1)
            if len(task_parts) == 2:
                task_id = task_parts[0].strip()
                task_name = task_parts[1].strip()
                prioritized_task_list.append(
                    {"task_id": task_id, "task_name": task_name})
        return prioritized_task_list

    def _get_top_tasks(self, vectorstore, query: str, k: int) -> List[str]:
        """Get the top k tasks based on the query."""
        results = vectorstore.similarity_search_with_score(query, k=k)
        if not results:
            return []
        sorted_results, _ = zip(
            *sorted(results, key=lambda x: x[1], reverse=True))
        return [str(item.metadata['task']) for item in sorted_results]

    def execute_task(self, vectorstore, execution_chain: LLMChain, objective: str, task: str, k: int = 5) -> str:
        """Execute a task."""
        context = self._get_top_tasks(vectorstore, query=objective, k=k)
        return execution_chain.run(objective=objective, context=context, task=task)

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs['objective']
        first_task = inputs.get("first_task", "Make a todo list")
        self.add_task({"task_id": 1, "task_name": first_task})
        num_iters = 0
        while True:
            if self.task_list:
                self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)

                # Step 2: Execute the task
                result = self.execute_task(
                    self.vectorstore, self.execution_chain, objective, task["task_name"]
                )
                this_task_id = int(task["task_id"])
                self.print_task_result(result)

                # Step 3: Store the result in Pinecone
                result_id = f"result_{task['task_id']}"
                self.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # Step 4: Create new tasks and reprioritize task list
                prioritized_task_list = self.create_and_prioritize_tasks(
                    self.task_management_chain, this_task_id, list(
                        self.task_list), result, task["task_name"], objective
                )
                for task in prioritized_task_list:
                    task_id = int(task["task_id"])
                    if task_id > self.task_id_counter:
                        self.task_id_counter = task_id
                        self.add_task(task)
                    else:
                        for existing_task in self.task_list:
                            if existing_task["task_id"] == task_id:
                                existing_task["task_name"] = task["task_name"]
                                break
                self.task_list = deque(prioritized_task_list)

            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print("\033[91m\033[1m" +
                      "\n*****TASK ENDING*****\n" + "\033[0m\033[0m")
                break
        return {}

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        vectorstore: VectorStore,
        verbose: bool = False,
        **kwargs
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_management_chain = TaskManagementChain.from_llm(
            llm, verbose=verbose)

        todo_prompt = PromptTemplate.from_template(
            "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}")
        todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
        search = SerpAPIWrapper()
        tools = [
            Tool(
                name="Search",
                func=search.run,
                description="useful for when you need to answer questions about current events"
            ),
            Tool(
                name="TODO",
                func=todo_chain.run,
                description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!"
            )
        ]

        executor_prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix="""You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}.""",
            suffix="""Question: {task}
            {agent_scratchpad}""",
            input_variables=["objective", "task",
                             "context", "agent_scratchpad"]
        )

        llm_chain = LLMChain(llm=llm, prompt=executor_prompt)
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True)
        return cls(
            task_management_chain=task_management_chain,
            execution_chain=agent_executor,
            vectorstore=vectorstore,
            **kwargs
        )


# *****************************
# Main process

# Configure OpenAI and Pinecone

# openai.api_key = OPENAI_API_KEY
# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
llm = OpenAI(temperature=0)

# Define your embedding model
embeddings_model = OpenAIEmbeddings()

# Create Pinecone index
# table_name = YOUR_TABLE_NAME
# dimension = 1536
# metric = "cosine"
# pod_type = "p1"
# if table_name not in pinecone.list_indexes():
#     pinecone.create_index(
#         table_name, dimension=dimension, metric=metric, pod_type=pod_type
#     )

# # Connect to the index
# index = pinecone.Index(table_name)

# Initialize the vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query,
                    index, InMemoryDocstore({}), {})

# Logging of LLMChains
verbose = False
# If None, will keep on going forever
# max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    verbose=verbose,
    # max_iterations=max_iterations
)
baby_agi({"objective": OBJECTIVE})
