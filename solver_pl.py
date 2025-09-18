import numpy as np
import matplotlib
matplotlib.use('Agg') # Usar backend no interactivo para evitar problemas de ventanas
import matplotlib.pyplot as plt
import tabulate # CORRECCIÓN 1: Importar el módulo completo
import sys
import subprocess

# --- Función del Método Simplex (con Tablas) ---

def metodo_simplex_completo(objective_function, constraints, problem_type):
    """
    Implementación del Método Simplex de Dos Fases para resolver problemas de PL.
    Muestra las tablas (tableau) en cada iteración.
    """
    num_vars = len(objective_function)
    
    # Construir la tabla inicial
    slack_vars_count = sum(1 for c in constraints if c['relation'] == '<=')
    surplus_vars_count = sum(1 for c in constraints if c['relation'] == '>=')
    artificial_vars_count = sum(1 for c in constraints if c['relation'] in ['>=', '='])
    
    num_total_vars = num_vars + slack_vars_count + surplus_vars_count + artificial_vars_count
    
    tableau = np.zeros((len(constraints) + 1, num_total_vars + 1))
    
    basis = [] # Almacena los índices de las variables básicas en cada fila

    s_idx, u_idx, a_idx = 0, 0, 0
    for i, const in enumerate(constraints):
        tableau[i, :num_vars] = const['coeffs']
        tableau[i, -1] = const['rhs']
        
        if const['relation'] == '<=':
            tableau[i, num_vars + s_idx] = 1
            basis.append(num_vars + s_idx)
            s_idx += 1
        elif const['relation'] == '>=':
            tableau[i, num_vars + slack_vars_count + u_idx] = -1
            tableau[i, num_vars + slack_vars_count + surplus_vars_count + a_idx] = 1
            basis.append(num_vars + slack_vars_count + surplus_vars_count + a_idx)
            u_idx += 1
            a_idx += 1
        elif const['relation'] == '=':
            tableau[i, num_vars + slack_vars_count + surplus_vars_count + a_idx] = 1
            basis.append(num_vars + slack_vars_count + surplus_vars_count + a_idx)
            a_idx += 1

    # Fase 1: Minimizar la suma de variables artificiales si existen
    if artificial_vars_count > 0:
        print("--- INICIANDO FASE 1 (Minimizar variables artificiales) ---")
        obj_fase1 = np.zeros(num_total_vars + 1)
        start_artificial = num_vars + slack_vars_count + surplus_vars_count
        obj_fase1[start_artificial : start_artificial + artificial_vars_count] = 1
        
        tableau[-1, :] = obj_fase1

        for i in range(len(constraints)):
            if basis[i] >= start_artificial:
                tableau[-1, :] -= tableau[i, :]
        
        tableau, basis = _resolver_simplex(tableau, basis, "Fase 1", 'minimizar')

        if abs(tableau[-1, -1]) > 1e-9:
            print("\n*** PROBLEMA INFACTIBLE ***")
            print("La Fase 1 terminó con un valor > 0. No existe una solución factible.")
            return
        
        tableau = np.delete(tableau, np.s_[start_artificial : start_artificial + artificial_vars_count], axis=1)
        basis = [b if b < start_artificial else -1 for b in basis]

    print("\n--- INICIANDO FASE 2 (Optimización del problema original) ---")
    
    obj_row = np.zeros(tableau.shape[1])
    if problem_type == 'maximizar':
        obj_row[:num_vars] = [-x for x in objective_function]
    else:
        obj_row[:num_vars] = objective_function
    
    tableau[-1, :] = obj_row

    for i, b_idx in enumerate(basis):
        if b_idx != -1 and tableau[-1, b_idx] != 0:
            tableau[-1, :] -= tableau[-1, b_idx] * tableau[i, :]
            
    tableau, basis = _resolver_simplex(tableau, basis, "Fase 2", problem_type)

    print("\n" + "="*25 + " CONCLUSIONES FINALES " + "="*25)
    if tableau is None: return

    solucion_optima = tableau[-1, -1]
    if problem_type == 'maximizar':
        print(f"✅ Valor Óptimo (Maximizado Z): {-solucion_optima:.4f}")
    else:
        print(f"✅ Valor Óptimo (Minimizado Z): {solucion_optima:.4f}")

    print("\nValores de las Variables de Decisión:")
    solucion = np.zeros(num_vars)
    for i in range(num_vars):
        if i in basis:
            row_idx = basis.index(i)
            solucion[i] = tableau[row_idx, -1]
    
    for i in range(num_vars):
        print(f"  - x{i+1} = {solucion[i]:.3f}")

def _resolver_simplex(tableau, basis, phase_name, problem_type):
    """Función auxiliar que itera para resolver el simplex."""
    iteration = 1
    
    while True:
        _print_tableau(tableau, iteration, phase_name, basis)
        
        obj_row = tableau[-1, :-1]

        if (problem_type == 'maximizar' and np.all(obj_row >= -1e-9)) or \
           (problem_type == 'minimizar' and np.all(obj_row <= 1e-9)):
            print(f"\n--- Óptimo encontrado en la {phase_name} ---")
            return tableau, basis

        if problem_type == 'maximizar':
            pivot_col = np.argmin(obj_row)
        else:
            pivot_col = np.argmax(obj_row)

        ratios = np.full(tableau.shape[0] - 1, np.inf)
        for i in range(tableau.shape[0] - 1):
            if tableau[i, pivot_col] > 1e-9:
                ratios[i] = tableau[i, -1] / tableau[i, pivot_col]
        
        if np.all(ratios == np.inf):
            print("\n*** PROBLEMA NO ACOTADO ***")
            print("La solución es infinita. No se puede determinar un óptimo.")
            return None, None
            
        pivot_row = np.argmin(ratios)

        print(f"-> Variable de entrada: Columna {pivot_col+1} | Variable de salida: Fila {pivot_row+1} (Variable básica en columna {basis[pivot_row]+1})")
        basis[pivot_row] = pivot_col

        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
        
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
        
        iteration += 1

def _print_tableau(tableau, iteration, phase_name, basis):
    """Imprime el tableau de forma legible."""
    print(f"\n--- {phase_name} | Iteración {iteration} ---")
    headers = [f"x{i+1}" for i in range(tableau.shape[1] - 1)] + ["Solución"]
    
    display_tableau = []
    for i in range(tableau.shape[0]-1):
        row_header = f"x{basis[i]+1}"
        display_tableau.append([row_header] + list(tableau[i,:]))
    
    display_tableau.append(["Z"] + list(tableau[-1,:]))

    # CORRECCIÓN 2: Llamar a la función a través del módulo
    print(tabulate.tabulate(display_tableau, headers=["Base"] + headers, floatfmt=".3f"))

def obtener_datos_usuario():
    """Guía al usuario para que ingrese los datos del problema de PL."""
    print("Bienvenido a la Herramienta de Programación Lineal")
    print("="*50)
    
    while True:
        problem_type = input("¿El problema es de maximizar o minimizar? (max/min): ").lower()
        if problem_type in ['max', 'maximizar', 'min', 'minimizar']:
            problem_type = 'maximizar' if problem_type.startswith('max') else 'minimizar'
            break
        print("Entrada inválida. Por favor, escribe 'max' o 'min'.")
        
    while True:
        try:
            obj_coeffs_str = input("Introduce los coeficientes de la función objetivo separados por comas (ej: 3,5,4): ")
            objective_function = [float(c.strip()) for c in obj_coeffs_str.split(',')]
            if not objective_function: raise ValueError
            break
        except ValueError:
            print("Entrada inválida. Asegúrate de introducir solo números separados por comas.")

    constraints = []
    i = 1
    while True:
        print(f"\n--- Restricción #{i} ---")
        while True:
            try:
                const_coeffs_str = input(f"Coeficientes de la restricción {i} (separados por comas): ")
                coeffs = [float(c.strip()) for c in const_coeffs_str.split(',')]
                if len(coeffs) != len(objective_function):
                    print(f"Error: Debes introducir {len(objective_function)} coeficientes.")
                    continue
                break
            except ValueError:
                print("Entrada inválida. Introduce números separados por comas.")

        while True:
            relation = input("Tipo de relación (<=, >=, =): ").strip()
            if relation in ['<=', '>=', '=']:
                break
            print("Entrada inválida. Usa '<=', '>=', o '='.")
            
        while True:
            try:
                rhs = float(input("Valor del lado derecho de la restricción (b): "))
                break
            except ValueError:
                print("Entrada inválida. Introduce un número.")

        constraints.append({'coeffs': coeffs, 'relation': relation, 'rhs': rhs})
        
        continuar = input("\n¿Añadir otra restricción? (s/n): ").lower()
        if continuar != 's':
            break
        i += 1
        
    return objective_function, constraints, problem_type

def metodo_grafico(objective_function, constraints, problem_type):
    """Resuelve un problema de programación lineal de dos variables y guarda el gráfico."""
    if len(objective_function) != 2:
        print("\nEl método gráfico solo es aplicable a problemas con 2 variables.")
        return

    print("\n--- Generando Gráfico ---")
    d = np.linspace(0, 150, 2000)
    x1, x2 = np.meshgrid(d, d)

    feasible_region = (x1 >= 0) & (x2 >= 0)

    for const in constraints:
        c1, c2 = const['coeffs']
        rel = const['relation']
        b = const['rhs']
        expr = c1*x1 + c2*x2
        if rel == '<=': feasible_region &= (expr <= b)
        elif rel == '>=': feasible_region &= (expr >= b)
        else: feasible_region &= np.isclose(expr, b)

    plt.figure(figsize=(8, 8))
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title("Método Gráfico: Región Factible (Gris) y Función Objetivo (Líneas)")
    
    plt.imshow(feasible_region.astype(int), 
                extent=(x1.min(), x1.max(), x2.min(), x2.max()),
                origin="lower", cmap="Greys", alpha=0.3)

    for const in constraints:
        c1, c2 = const['coeffs']
        b = const['rhs']
        if abs(c2) > 1e-6:
            y_vals = (b - c1 * d) / c2
            plt.plot(d, y_vals, label=f'{c1}x1 + {c2}x2 = {b}')
        elif abs(c1) > 1e-6:
             plt.axvline(x=b/c1, label=f'{c1}x1 = {b}')

    Z = objective_function[0]*x1 + objective_function[1]*x2
    contours = plt.contour(x1, x2, Z, 15, cmap='viridis')
    plt.clabel(contours, inline=True, fontsize=8)

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    
    plt.savefig('grafico_solucion.png')
    print("\n✅ Gráfico guardado como 'grafico_solucion.png' en tu carpeta.")


# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":
    try:
        import tabulate
    except ImportError:
        print("La biblioteca 'tabulate' no está instalada. Intentando instalar...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
        except Exception as e:
            print(f"Error al instalar tabulate: {e}")
            print("Por favor, instálala manualmente con: pip install tabulate")
            sys.exit(1)

    objective_function, constraints, problem_type = obtener_datos_usuario()
    
    metodo_simplex_completo(objective_function, constraints, problem_type)
    
    metodo_grafico(objective_function, constraints, problem_type)