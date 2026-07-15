"""Execute the frozen ArticleV1 minimal real-agent experiment."""

import json
from pathlib import Path

from trustworthy_agent.article_v1.minimal_e2_execution import execute

if __name__ == "__main__":
    print(json.dumps(execute(Path.cwd()), indent=2, sort_keys=True))
