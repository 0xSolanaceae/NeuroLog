#!/usr/bin/env python3

import argparse
import logging
import os
from yaspin import yaspin
from analyzer import LogAnalyzer

def cli_main():
    parser = argparse.ArgumentParser(
        description="ML-Powered Log Analysis CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    analyze_parser = add_subcommand(
        subparsers,
        'analyze',
        'Run full log analysis',
        "Output file for anomalies"
    )
    analyze_parser.add_argument(
        "--format", "-f", 
        choices=['csv', 'json', 'html'], 
        default='csv', 
        help="Output format"
    )

    train_parser = subparsers.add_parser('train-crf', help='Train CRF model for log parsing')
    train_parser.add_argument("--output", "-o", default="crf_model.pkl", 
                              help="Output path for trained model")

    args = parser.parse_args()

    if args.command == 'analyze' and not os.path.isfile(args.log_file):
        logging.error("File not found: %s", args.log_file)
        exit(1)

    analyzer = LogAnalyzer()

    if args.command == 'analyze':
        analysis_output(analyzer, args)
    elif args.command == 'train-crf':
        from train_crf import train_and_save_model
        with yaspin(text="Training CRF model", color="yellow") as spinner:
            train_and_save_model(args.output)
            spinner.ok("✔")

def analysis_output(analyzer, args):
    with yaspin(text="Analyzing log file", color="cyan") as spinner:
        results = analyzer.analyze(args.log_file)
        spinner.ok("✔")
    print("\n=== ANALYSIS RESULTS ===")
    print("Log Type Confidences:")
    for t, c in results['type_confidences'].items():
        print(f"- {t.title()}: {c:.1%}")
    print("\nTop Anomalies:")
    print(results['anomalies'][['timestamp', 'host', 'message']].head(10).to_string(index=False))
    print("\nStatistics:")
    print(f"Total Entries: {results['stats']['total_entries']}")
    print(f"Error Rate: {results['stats']['error_rate']:.1%}")
    if args.output:
        if args.format == 'csv':
            results['anomalies'].to_csv(args.output, index=False)
        elif args.format == 'json':
            results['anomalies'].to_json(args.output, orient='records')
        elif args.format == 'html':
            results['anomalies'].to_html(args.output)
        logging.info("Results saved to %s", args.output)
        print(f"\nResults saved to {args.output}")

def add_subcommand(subparsers, name, help_text, output_help):
    result = subparsers.add_parser(name, help=help_text)
    result.add_argument("log_file", help="Path to the log file")
    result.add_argument("--output", "-o", help=output_help)
    return result

if __name__ == '__main__':
    cli_main()