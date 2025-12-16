""" Módulo de interfaz de usuario por linea de comandos. """

import os


def clear_screen():
    """Limpia la pantalla de la terminal"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Imprime el encabezado del sistema"""
    print("=" * 60)
    print("   SISTEMA DE CLASIFICADORES")
    print("=" * 60)
    print()


def print_main_menu():
    """Imprime el menú principal"""
    print("MENÚ PRINCIPAL")
    print("-" * 50)
    print()
    print("  [1] Preprocesamiento")
    print("  [2] Entrenamiento")
    print("  [3] Pruebas / Clasificación en tiempo real")
    print()
    print("  [0] Salir")
    print()


def print_preprocessing_menu():
    """Imprime el submenú de preprocesamiento"""
    print("PREPROCESAMIENTO")
    print("-" * 50)
    print()
    print("  [1] Dividir dataset (train/val/test)")
    print("  [2] Aumentar dataset (data augmentation)")
    print()
    print("  [0] Volver al menú principal")
    print()


def print_training_menu(architectures: dict):
    """
    Imprime el submenú de entrenamiento dinámicamente.
    
    Args:
        architectures: Diccionario con las arquitecturas disponibles
    """
    print("ENTRENAMIENTO")
    print("-" * 50)
    print()
    for i, (key, arch) in enumerate(architectures.items(), 1):
        print(f"  [{i}] Entrenar con {arch['name']}")
    print()
    print("  [0] Volver al menú principal")
    print()


def print_testing_menu(architectures: dict):
    """
    Imprime el submenú de pruebas dinámicamente.
    
    Args:
        architectures: Diccionario con las arquitecturas disponibles
    """
    print("PRUEBAS / CLASIFICACIÓN EN TIEMPO REAL")
    print("-" * 50)
    print()
    for i, (key, arch) in enumerate(architectures.items(), 1):
        print(f"  [{i}] Clasificación en tiempo real ({arch['name']})")
    print()
    print("  [0] Volver al menú principal")
    print()


def print_action_header(message: str):
    """Imprime encabezado para una acción"""
    print(f"\n{message}")
    print("-" * 40)


def print_error(message: str = "Opción no válida. Intenta de nuevo."):
    """Imprime mensaje de error"""
    print(f"\n{message}")


def wait_for_enter(message: str = "\nPresiona Enter para continuar..."):
    """Espera a que el usuario presione Enter"""
    input(message)
