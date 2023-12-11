from .imports import *

def check_install(path):
    ppa = PPA(path)
    passed = True
    if not os.path.exists(ppa.path_metadata):
        click.echo(f'No metadata file found at {ppa.path_metadata}')
        passed = False
    if not os.path.exists(ppa.path_pages_jsonl):
        click.echo(f'No page jsonl file found at {ppa.path_pages_jsonl}')
        passed = False
    click.echo(f"Checks {'failed' if not passed else 'passed'}")
    return passed
    
@click.group()
def cli(): pass

@cli.command()
@click.argument('command', type=click.Choice(['check','preproc','gen_db']))
@click.option('--path', default=PATH_PPA_CORPUS, help=f'path to corpus [default: {PATH_PPA_CORPUS}]')
def run(command, path):
    if command=='check':
        click.echo(f'Checking installation...')
        check_install(path)
    