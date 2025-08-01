from src.Starter import Starter

if __name__ == '__main__':
    while True:
        show_plots = input("Should the plots be displayed? (yes/no): ").strip().lower()
        if show_plots in ("yes", "no"):
            break
        print("Please enter 'yes' or 'no'.\n")

    starter = Starter(show_plots == "yes")
    starter.start()

