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
python scripts/build.py --root_dir ROOT_DIR --platform {win,linux,mac} --mode {dev,prod} --no-cache --no-packing --output_dir OUTPUT_DIR

Example:
python scripts/build.py --root_dir . --platform win --mode dev --no-packing --output_dir dist
python scripts/build.py --root_dir . --platform mac --mode dev --no-packing --incremental --output_dir dist
python scripts/build.py --root_dir . --platform win --mode prod --no-cache --output_dir dist

### Full build
If you have made any changes to the requirements.txt file or you have modified the main.py file in the root directory itself, you need perform a full build. 

### Incremental build
For other project file changes under the app/ and plugins/ directories, you can perform a faster incremental build:
python scripts/build.py --root_dir . --platform win --mode dev --no-packing --incremental --output_dir dist


## License

Copyright © 2025 Synvo AI. All rights reserved.

This software is proprietary and confidential. Unauthorized copying, distribution, or use of this software, via any medium, is strictly prohibited without the prior written permission of Synvo AI.

For licensing inquiries, please contact us through [www.synvo.ai](https://www.synvo.ai).

See our [Terms of Service](https://www.synvo.ai/terms) for more details.