import argparse
from train import train_agent
from evaluate import evaluate_agent
from run_agent import run_agent
from interactive_run import interactive_run_agent
import sys

def main():
    """
    Script principale per gestire i comandi di addestramento, valutazione ed esecuzione degli agenti.
    """
    parser = argparse.ArgumentParser(description="Progetto di Navigazione Indoor con RL")
    subparsers = parser.add_subparsers(dest='command', help='Comandi disponibili')

    # Comando per addestrare gli agenti
    parser_train = subparsers.add_parser('train', help='Addestra gli agenti')
    parser_train.add_argument('--agent', type=str, choices=['sarsa', 'q_learning', 'all'], default='all',
                              help='Tipo di agente da addestrare')
    parser_train.add_argument('--episodes', type=int, default=50000, help='Numero di episodi di addestramento')
    parser_train.add_argument('--grid_size', type=int, default=15, help='Dimensione della griglia')
    parser_train.add_argument('--num_obstacles', type=int, default=10, help='Numero di ostacoli')
    parser_train.add_argument('--min_distance', type=int, default=2, help='Distanza minima tra ostacoli e tra ostacoli e agenti/target')
    parser_train.add_argument('--fixed_obstacles', type=int, nargs='+', help='Posizioni fisse per ostacoli (es. x1 y1 x2 y2 ...)')

    # Comando per valutare gli agenti
    parser_evaluate = subparsers.add_parser('evaluate', help='Valuta gli agenti addestrati')
    parser_evaluate.add_argument('--agent', type=str, choices=['sarsa', 'q_learning', 'all'], default='all',
                                 help='Tipo di agente da valutare')
    parser_evaluate.add_argument('--episodes', type=int, default=100, help='Numero di episodi di valutazione')
    parser_evaluate.add_argument('--grid_size', type=int, default=15, help='Dimensione della griglia')
    parser_evaluate.add_argument('--num_obstacles', type=int, default=10, help='Numero di ostacoli')
    parser_evaluate.add_argument('--min_distance', type=int, default=2, help='Distanza minima tra ostacoli e tra ostacoli e agenti/target')
    parser_evaluate.add_argument('--fixed_obstacles', type=int, nargs='+', help='Posizioni fisse per ostacoli (es. x1 y1 x2 y2 ...)')

    # Comando per eseguire un agente addestrato in modalità non interattiva
    parser_run = subparsers.add_parser('run', help='Esegui un agente addestrato')
    parser_run.add_argument('--agent', type=str, choices=['sarsa', 'q_learning'], required=True,
                            help='Tipo di agente da eseguire')
    parser_run.add_argument('--episodes', type=int, default=1, help='Numero di episodi da eseguire')
    parser_run.add_argument('--grid_size', type=int, default=15, help='Dimensione della griglia')
    parser_run.add_argument('--num_obstacles', type=int, default=10, help='Numero di ostacoli')
    parser_run.add_argument('--min_distance', type=int, default=2, help='Distanza minima tra ostacoli e tra ostacoli e agenti/target')
    parser_run.add_argument('--render', action='store_true', help='Visualizza la griglia durante l\'esecuzione')
    parser_run.add_argument('--fixed_obstacles', type=int, nargs='+', help='Posizioni fisse per ostacoli (es. x1 y1 x2 y2 ...)')

    # Comando per eseguire un agente in modalità interattiva
    parser_interactive = subparsers.add_parser('interactive_run', help='Esegui un agente addestrato in modalità interattiva')
    parser_interactive.add_argument('--agent', type=str, choices=['sarsa', 'q_learning'], required=True,
                                    help='Tipo di agente da eseguire')
    parser_interactive.add_argument('--grid_size', type=int, default=15, help='Dimensione della griglia')
    parser_interactive.add_argument('--num_obstacles', type=int, default=10, help='Numero di ostacoli')
    parser_interactive.add_argument('--min_distance', type=int, default=2, help='Distanza minima tra ostacoli e tra ostacoli e agenti/target')
    parser_interactive.add_argument('--fixed_obstacles', type=int, nargs='+', help='Posizioni fisse per ostacoli (es. x1 y1 x2 y2 ...)')

    # Parsing degli argomenti da linea di comando
    args = parser.parse_args()

    # Conversione degli ostacoli fissi se forniti
    fixed_obstacles = None
    if args.fixed_obstacles:
        if len(args.fixed_obstacles) % 2 != 0:
            print("Errore: Le posizioni degli ostacoli fissi devono essere specificate come coppie x y.")
            sys.exit(1)
        fixed_obstacles = [args.fixed_obstacles[i:i+2] for i in range(0, len(args.fixed_obstacles), 2)]

    # Gestione dei comandi
    if args.command == 'train':
        if args.agent in ['sarsa', 'all']:
            print("Addestramento agente SARSA...")
            train_agent(
                agent_type='sarsa',
                episodes=args.episodes,
                grid_size=args.grid_size,
                num_obstacles=args.num_obstacles,
                fixed_obstacles=fixed_obstacles,
                min_distance=args.min_distance
            )
        if args.agent in ['q_learning', 'all']:
            print("\nAddestramento agente Q-Learning...")
            train_agent(
                agent_type='q_learning',
                episodes=args.episodes,
                grid_size=args.grid_size,
                num_obstacles=args.num_obstacles,
                fixed_obstacles=fixed_obstacles,
                min_distance=args.min_distance
            )
    elif args.command == 'evaluate':
        if args.agent in ['sarsa', 'all']:
            print("Valutazione agente SARSA...")
            evaluate_agent(
                agent_type='sarsa',
                episodes=args.episodes,
                grid_size=args.grid_size,
                num_obstacles=args.num_obstacles,
                fixed_obstacles=fixed_obstacles,
                min_distance=args.min_distance
            )
        if args.agent in ['q_learning', 'all']:
            print("\nValutazione agente Q-Learning...")
            evaluate_agent(
                agent_type='q_learning',
                episodes=args.episodes,
                grid_size=args.grid_size,
                num_obstacles=args.num_obstacles,
                fixed_obstacles=fixed_obstacles,
                min_distance=args.min_distance
            )
    elif args.command == 'run':
        print(f"Esecuzione agente {args.agent}...")
        run_agent(
            agent_type=args.agent,
            episodes=args.episodes,
            grid_size=args.grid_size,
            num_obstacles=args.num_obstacles,
            fixed_obstacles=fixed_obstacles,
            min_distance=args.min_distance,
            render=args.render
        )
    elif args.command == 'interactive_run':
        print(f"Esecuzione agente {args.agent} in modalità interattiva...")
        interactive_run_agent(
            agent_type=args.agent,
            grid_size=args.grid_size,
            num_obstacles=args.num_obstacles,
            fixed_obstacles=fixed_obstacles,
            min_distance=args.min_distance
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
