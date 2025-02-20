#!/usr/bin/env python3

import argparse
import logging
import os
import pandas as pd
from yaspin import yaspin
from analyzer import LogAnalyzer

def cli_main():
    parser = argparse.ArgumentParser(
        description="ML-Powered Log Analysis CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    analyze_parser = parsers(
        subparsers,
        'analyze',
        'Run full log analysis',
        "Output file for anomalies",
    )
    analyze_parser.add_argument(
        "--format", "-f", 
        choices=['csv', 'json', 'html'], 
        default='csv', 
        help="Output format"
    )

    # Subcommand: detect
    detect_parser = subparsers.add_parser('detect', help='Detect log formats')
    detect_parser.add_argument("log_file", help="Path to the log file")

    parsers(
        subparsers, 'stats', 'Generate log statistics', "Output file for stats"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.log_file):
        logging.error("File not found: %s", args.log_file)
        exit(1)

    analyzer = LogAnalyzer()

    if args.command == 'analyze':
        analysis_output(analyzer, args)
    elif args.command == 'detect':
        confidences = analyzer.detect_log_types(args.log_file)
        print("Log Format Probabilities:")
        for t, p in sorted(confidences.items(), key=lambda x: x[1], reverse=True):
            print(f"- {t:<12}: {p:.1%}")

    elif args.command == 'stats':
        stats_output(analyzer, args)


def stats_output(analyzer, args):
    results = analyzer.analyze(args.log_file)
    stats = results.get('stats', {})
    print("\n=== STATISTICS ===")
    print(f"Total Entries: {stats.get('total_entries', 0)}")
    print(f"Error Rate: {stats.get('error_rate', 0):.1%}")
    print(f"Common Hosts: {stats.get('common_hosts', {})}")
    print("\nHTTP Status Distribution:")
    print(stats.get('http_status_distribution', {}))
    if args.output:
        pd.Series(stats).to_json(args.output)
        logging.info("Statistics saved to %s", args.output)
        print(f"Statistics saved to {args.output}")


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


def parsers(subparsers, arg1, help, arg3):
    # Subcommand: analyze
    result = subparsers.add_parser(arg1, help=help)
    result.add_argument("log_file", help="Path to the log file")
    result.add_argument("--output", "-o", help=arg3)
    return result

if __name__ == '__main__':
    cli_main()
