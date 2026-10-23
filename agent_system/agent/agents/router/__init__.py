# Copyright 2026 Nanyang Technological University (NTU), Singapore
# Copyright 2026 Dr. MAS Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from agent_system.agent.registry import AgentRegistry

from .router_agent import RouterAgent
from .math import SolverAgent, ReviserAgent, VerifierAgent as MathVerifierAgent
from .search import SearchAgent, AnswerAgent, VerifierAgent as SearchVerifierAgent


@AgentRegistry.register("VERIFY")
def create_verifier_agent(wg_id, tokenizer, processor, config):
    """Keep one Router role name while selecting the scene's verifier class."""
    verifier_class = {"math": MathVerifierAgent, "search": SearchVerifierAgent}[config.env.env_name]
    return verifier_class(wg_id=wg_id, tokenizer=tokenizer, processor=processor, config=config)


__all__ = [
    "RouterAgent",
    "SolverAgent",
    "MathVerifierAgent",
    "ReviserAgent",
    "SearchAgent",
    "SearchVerifierAgent",
    "AnswerAgent",
]
