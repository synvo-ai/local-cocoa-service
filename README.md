# Synvo File System

## Overview

`local-cocoa-server` is the backend service for local-cocoa, offering convenient APIs.

---

## Build Guild

1. Rename .env.prod.example to .env.prod and then Add your configuration to .env.prod

2. python -m venv .venv

3. activate your .venv

4. pip install -r app/requirements.txt

5. build command:
python scripts/build.py --root_dir ROOT_DIR --platform {win,linux,macos} --mode {dev,prod} --no-cache {true, false} --output_dir OUTPUT_DIR

Example:
python scripts/build.py --root_dir . --platform win --mode prod --no-cache true --output_dir dist

## License

Copyright © 2025 Synvo AI. All rights reserved.

This software is proprietary and confidential. Unauthorized copying, distribution, or use of this software, via any medium, is strictly prohibited without the prior written permission of Synvo AI.

For licensing inquiries, please contact us through [www.synvo.ai](https://www.synvo.ai).

See our [Terms of Service](https://www.synvo.ai/terms) for more details.