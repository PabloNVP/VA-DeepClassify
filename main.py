""" Sistema de Clasificadores """

from src.ui.menus import (
    clear_screen, print_header, print_main_menu,
    print_preprocessing_menu, print_training_menu, print_testing_menu,
    print_action_header, print_error, wait_for_enter
)
from src.training.architectures import ARCHITECTURES

def run_preprocessing_option(option):
    """Ejecuta la opción de preprocesamiento seleccionada"""
    if option == "1":
        print_action_header("Ejecutando división de dataset...")
        from src.preprocessing.split_dataset import split_dataset
        split_dataset()
        wait_for_enter()
    elif option == "2":
        print_action_header("Ejecutando aumento de datos...")
        from src.preprocessing.augment_dataset import augment_dataset
        augment_dataset()
        wait_for_enter()


def run_training_option(option):
    """Ejecuta la opción de entrenamiento seleccionada"""
    from src.training.train_model import train_model
    
    # Generar mapeo dinámico
    option_to_arch = {str(i): key for i, key in enumerate(ARCHITECTURES.keys(), 1)}
    
    if option in option_to_arch:
        architecture = option_to_arch[option]
        arch_name = ARCHITECTURES[architecture]["name"]
        print_action_header(f"Iniciando entrenamiento con {arch_name}...")
        train_model(architecture=architecture)
        wait_for_enter()


def run_testing_option(option):
    """Ejecuta la opción de pruebas seleccionada"""
    from src.testing.realtime_classification import run_realtime_classification
    
    # Generar mapeo dinámico
    option_to_arch = {str(i): key for i, key in enumerate(ARCHITECTURES.keys(), 1)}
    
    if option in option_to_arch:
        architecture = option_to_arch[option]
        arch_name = ARCHITECTURES[architecture]["name"]
        print_action_header(f"Iniciando clasificación en tiempo real ({arch_name})...")
        run_realtime_classification(architecture=architecture)
        wait_for_enter()


def preprocessing_submenu():
    """Submenú de preprocesamiento"""
    valid_options = ["1", "2"]
    while True:
        clear_screen()
        print_header()
        print_preprocessing_menu()
        
        option = input("Selecciona una opción: ").strip()
        
        if option == "0":
            break
        elif option in valid_options:
            run_preprocessing_option(option)
        else:
            print_error()
            wait_for_enter()


def training_submenu():
    """Submenú de entrenamiento"""
    valid_options = [str(i) for i in range(1, len(ARCHITECTURES) + 1)]
    while True:
        clear_screen()
        print_header()
        print_training_menu(ARCHITECTURES)
        
        option = input("Selecciona una opción: ").strip()
        
        if option == "0":
            break
        elif option in valid_options:
            run_training_option(option)
        else:
            print_error()
            wait_for_enter()


def testing_submenu():
    """Submenú de pruebas"""
    valid_options = [str(i) for i in range(1, len(ARCHITECTURES) + 1)]
    while True:
        clear_screen()
        print_header()
        print_testing_menu(ARCHITECTURES)
        
        option = input("Selecciona una opción: ").strip()
        
        if option == "0":
            break
        elif option in valid_options:
            run_testing_option(option)
        else:
            print_error()
            wait_for_enter()


def main():
    """Función principal del programa"""
    while True:
        clear_screen()
        print_header()
        print_main_menu()
        
        option = input("Selecciona una opción: ").strip()
        
        if option == "0":
            print("\n¡Hasta luego!")
            break
        elif option == "1":
            preprocessing_submenu()
        elif option == "2":
            training_submenu()
        elif option == "3":
            testing_submenu()
        else:
            print_error()
            wait_for_enter()


if __name__ == "__main__":
    main()
