#!/usr/bin/env python3

import logging
from cli import cli_main 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    cli_main()