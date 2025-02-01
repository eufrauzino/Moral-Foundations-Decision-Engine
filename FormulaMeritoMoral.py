import os
import numpy as np
import matplotlib.pyplot as plt
import random
import openai
from concurrent.futures import ProcessPoolExecutor
from deap import base, creator, tools, algorithms

# Configurar a API do OpenAI (opcional – defina a variável de ambiente OPENAI_API_KEY)
openai.api_key = os.getenv('OPENAI_API_KEY')


###############################
# Funções Auxiliares Gerais
###############################

def normalize(vector):
    """Normaliza um vetor para que sua norma seja 1."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("A função de onda não pode ser o vetor nulo.")
    return vector / norm


def is_hermitian(matrix):
    """Verifica se uma matriz é Hermitiana (igual à sua transposta conjugada)."""
    return np.allclose(matrix, matrix.conj().T)


def make_hermitian(matrix):
    """Força a hermiticidade ajustando a matriz: (M + M†)/2."""
    return (matrix + matrix.conj().T) / 2


###############################
# Funções para Geração Automática de Operadores
###############################

def get_operator_chatgpt(num_states, factor_name, states, factor_database):
    """
    Gera automaticamente o operador para um fator usando a API do ChatGPT.
    Se a API não estiver disponível ou o fator não estiver no banco de dados,
    utiliza valores padrão ou gera valores aleatórios.
    O operador é construído como uma matriz diagonal.
    """
    print(f"\nGerando operador para o fator '{factor_name}' automaticamente.")

    if openai.api_key and (factor_name.lower() in factor_database):
        prompt = f"Para os estados de decisão: {', '.join(states)}, atribua um valor numérico de 1 a 10 para o fator '{factor_name}'. " \
                 f"Retorne os valores no formato: Estado 1: <valor1>, Estado 2: <valor2>."
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "Você é um assistente que atribui valores numéricos de forma objetiva."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=60,
                temperature=0.5,
            )
            reply = response.choices[0].message.content.strip()
            factor_values = {}
            for part in reply.split(','):
                if ':' in part:
                    state_label, value_str = part.split(':')
                    try:
                        num = int(''.join(filter(str.isdigit, state_label)))
                    except ValueError:
                        continue
                    state_name = states[num - 1] if num - 1 < len(states) else f"Estado{num}"
                    factor_values[state_name] = float(value_str.strip())
            if factor_values:
                print(f"Valores gerados para '{factor_name}': {factor_values}")
            else:
                raise Exception("Resposta não interpretada.")
        except Exception as e:
            print(f"Erro com ChatGPT: {e}")
            factor_values = None
    else:
        factor_values = None

    if not factor_values:
        if factor_name.lower() in factor_database:
            factor_values = factor_database[factor_name.lower()]
            print(f"Usando valores padrão do banco de dados para '{factor_name}': {factor_values}")
        else:
            factor_values = {state: random.uniform(1, 10) for state in states}
            print(f"Fator '{factor_name}' não encontrado. Usando valores aleatórios: {factor_values}")

    # Construir o operador como uma matriz diagonal
    operator = np.zeros((num_states, num_states), dtype=complex)
    for i in range(num_states):
        state = states[i]
        value = factor_values.get(state, random.uniform(1, 10))
        operator[i, i] = value
    if not is_hermitian(operator):
        operator = make_hermitian(operator)
    return operator


def calculate_uncertainty(psi, operator):
    """
    Calcula o valor esperado e a incerteza (desvio padrão) de um operador dado o estado psi.
    Usa: σ = sqrt(<psi|F^2|psi> - (<psi|F|psi>)^2)
    """
    expected_value = np.vdot(psi, operator @ psi).real
    expected_value_squared = np.vdot(psi, operator @ operator @ psi).real
    variance = expected_value_squared - expected_value ** 2
    uncertainty = np.sqrt(max(variance, 0))
    return expected_value, uncertainty


###############################
# Funções para Questionário Moral (Baseado no MFQ)
###############################

def get_moral_profile():
    """
    Aplica um breve questionário baseado no MFQ para captar o perfil moral do usuário.
    Retorna um dicionário com as pontuações dos fundamentos.
    """
    print("\nResponda as seguintes perguntas (escala de 1 a 10):")
    questions = {
        "harm": "Quão importante é evitar causar danos aos outros?",
        "fairness": "Quão importante é tratar todos de forma justa?",
        "loyalty": "Quão importante é manter a lealdade ao seu grupo?",
        "authority": "Quão importante é respeitar as autoridades e regras?",
        "purity": "Quão importante é manter a pureza moral e os valores tradicionais?"
    }
    profile = {}
    for key, question in questions.items():
        while True:
            try:
                response = float(input(f"{question} "))
                if response < 1 or response > 10:
                    print("Por favor, insira um número entre 1 e 10.")
                    continue
                profile[key] = response
                break
            except ValueError:
                print("Entrada inválida. Tente novamente.")
    return profile


def adjust_weights_by_profile(base_weights, moral_profile, factor_to_moral):
    """
    Ajusta os pesos dos fatores com base no perfil moral do usuário.
    base_weights: dicionário com os pesos base.
    moral_profile: dicionário com as pontuações dos fundamentos morais.
    factor_to_moral: mapeamento de cada fator para uma lista de fundamentos relevantes.
    Retorna os pesos ajustados (normalizados para soma 1).
    """
    adjusted_weights = {}
    for factor in base_weights:
        if factor.lower() in factor_to_moral:
            morals = factor_to_moral[factor.lower()]
            score = np.mean([moral_profile.get(m, 5) for m in morals])
        else:
            score = 5  # neutro
        # Mapeia score de [1,10] para ajuste de [0.5, 1.5]
        adjustment = 0.5 + (score - 1) * (1.0 / 9)
        adjusted_weights[factor] = base_weights[factor] * adjustment
    total = sum(adjusted_weights.values())
    for factor in adjusted_weights:
        adjusted_weights[factor] /= total
    return adjusted_weights


###############################
# Funções para Simulação Monte Carlo Paralela
###############################

def simulate_single(psi, operators, weights, k_value, interactions, uncertainties):
    """
    Executa uma única simulação Monte Carlo para calcular M(A).
    Cada fator é amostrado de uma distribuição normal com média e desvio (incerteza).
    """
    M_A_sim = 0.0
    for factor in operators:
        exp_val, uncert = calculate_uncertainty(psi, operators[factor])
        sampled_val = np.random.normal(exp_val, uncert)
        M_A_sim += weights[factor] * sampled_val
    penalty = k_value * sum([weights[factor] * uncertainties[factor] for factor in operators])
    M_A_sim -= penalty
    for inter in interactions.values():
        M_A_sim += inter['theta'] * inter['expected_product']
    return M_A_sim


def monte_carlo_consensus_parallel(psi, operators, weights, k_value, interactions, uncertainties, num_simulations=1000):
    """
    Executa simulações Monte Carlo em paralelo para calcular o consenso final de M(A).
    Retorna:
      - consensus: média dos valores simulados.
      - probability: fração das simulações em que M(A) > 0.
      - results: array com os valores simulados.
    """
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(simulate_single, psi, operators, weights, k_value, interactions, uncertainties)
                   for _ in range(num_simulations)]
        results = np.array([future.result() for future in futures])
    consensus = np.mean(results)
    probability = np.sum(results > 0) / num_simulations
    return consensus, probability, results


###############################
# Função de Otimização Evolutiva (opcional, usando DEAP)
###############################

def genetic_optimization(psi, operators, base_weights, k_value, interactions, uncertainties, num_simulations=500,
                         num_generations=20, pop_size=50):
    """
    Otimiza os ajustes (multiplicadores) para os pesos dos fatores usando um algoritmo genético.
    Retorna o melhor vetor de ajustes e seu fitness (a probabilidade de M(A) > 0).
    """
    num_factors = len(operators)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # Cada gene (ajuste) varia entre 0.5 e 1.5
    toolbox.register("attr_float", random.uniform, 0.5, 1.5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_factors)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_individual(individual):
        # Calcular pesos ajustados: w_eff = base_weight * ajuste, normalizados
        factor_list = list(operators.keys())
        raw_weights = np.array([base_weights[f] for f in factor_list])
        adjustments = np.array(individual)
        new_weights = raw_weights * adjustments
        new_weights /= new_weights.sum()
        adjusted_weights = {factor_list[i]: new_weights[i] for i in range(len(factor_list))}
        consensus, probability, _ = monte_carlo_consensus_parallel(psi, operators, adjusted_weights, k_value,
                                                                   interactions, uncertainties, num_simulations)
        return (probability,)

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=pop_size)
    NGEN = num_generations
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        fits = list(map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(offspring))
        best = tools.selBest(population, 1)[0]
        print(f"Geração {gen + 1}: Melhor fitness = {best.fitness.values[0]:.4f}")
    best = tools.selBest(population, 1)[0]
    return best, best.fitness.values[0]


###############################
# Função Principal com Feedback Iterativo
###############################

def main():
    print("Bem-vindo ao Sistema Híbrido de Tomada de Decisão com Otimização Evolutiva e Paralelismo!\n")

    # Entrada dos estados de decisão (em uma única linha, separados por vírgulas)
    states_input = input("Digite os estados de decisão (separados por vírgula): ")
    states = [s.strip() for s in states_input.split(',') if s.strip()]
    num_states = len(states)
    if num_states == 0:
        raise ValueError("Nenhum estado informado.")

    # Definir amplitudes iguais para reduzir viés
    psi = np.ones(num_states, dtype=complex)
    psi = normalize(psi)

    # Obter o perfil moral do usuário (questionário baseado no MFQ)
    moral_profile = get_moral_profile()
    print("\nSeu perfil moral:")
    for key, val in moral_profile.items():
        print(f"{key}: {val}")

    # Entrada dos fatores (em uma única linha, separados por vírgula)
    factors_input = input("Digite os nomes dos fatores (separados por vírgula): ")
    factor_names = [s.strip() for s in factors_input.split(',') if s.strip()]
    num_factors = len(factor_names)
    if num_factors == 0:
        raise ValueError("Nenhum fator informado.")

    # Mapeamento entre fatores e fundamentos morais para ajuste de pesos
    factor_to_moral = {
        "descanso": ["harm"],
        "produtividade": ["fairness"],
        "saúde": ["harm", "purity"],
        "bemestarsocial": ["loyalty", "authority"]
    }

    # Banco de dados interno de valores padrão para alguns fatores
    factor_database = {
        "descanso": {"Dormir Agora": 3, "Dormir Mais Tarde": 5},
        "produtividade": {"Dormir Agora": 2, "Dormir Mais Tarde": 4},
        "saúde": {"Dormir Agora": 4, "Dormir Mais Tarde": 6},
        "bemestarsocial": {"Dormir Agora": 1, "Dormir Mais Tarde": 3},
        "custo": {"Mudar de casa": 6, "Reformar a casa" :4},
        "conforto":{"Mudar de casa": 8, "Reformar a casa" :6},
        "localizacao": {"Mudar de casa": 9, "Reformar a casa": 5}
    }

    # Gerar operadores automaticamente para cada fator
    operators = {}
    expected_values = {}
    uncertainties = {}
    for factor_name in factor_names:
        op = get_operator_chatgpt(num_states, factor_name, states, factor_database)
        exp_val, uncert = calculate_uncertainty(psi, op)
        operators[factor_name] = op
        expected_values[factor_name] = exp_val
        uncertainties[factor_name] = uncert

    # Cálculo dos pesos base inversamente proporcionais às variâncias
    variances = {k: uncertainties[k] ** 2 for k in uncertainties}
    total_inverse_variance = sum([1 / v if v > 0 else 0 for v in variances.values()])
    base_weights = {}
    for k in variances:
        base_weights[k] = (1 / variances[k]) / total_inverse_variance if variances[k] > 0 else 0
    total_weight = sum(base_weights.values())
    if total_weight == 0:
        raise ValueError("A soma dos pesos é zero. Verifique as variâncias dos fatores.")
    for k in base_weights:
        base_weights[k] /= total_weight

    print("\nPesos base dos fatores (calculados inversamente às variâncias):")
    for k in base_weights:
        print(f"Fator '{k}': Peso base = {base_weights[k]:.4f}")

    # Ajuste dos pesos com base no perfil moral do usuário
    adjusted_weights = adjust_weights_by_profile(base_weights, moral_profile, factor_to_moral)
    print("\nPesos ajustados com base no seu perfil moral:")
    for k in adjusted_weights:
        print(f"Fator '{k}': Peso ajustado = {adjusted_weights[k]:.4f}")

    # Definir valor de k (nível de confiança) fixo para simplificar
    k_value = 1  # Ex: 1 para 68% de confiança

    # Não definimos interações manualmente nesta versão para simplificar
    interactions = {}

    # Executar simulações Monte Carlo em paralelo para obter consenso e probabilidade
    num_simulations = int(input("\nDigite o número de simulações Monte Carlo (ex.: 1000): "))
    consensus, probability, simulation_results = monte_carlo_consensus_parallel(psi, operators, adjusted_weights,
                                                                                k_value, interactions, uncertainties,
                                                                                num_simulations)

    print(f"\nResultado do Consenso Monte Carlo:")
    print(f"Consenso M(A): {consensus:.4f}")
    print(f"Probabilidade de decisão favorável (M(A) > 0): {probability * 100:.2f}%")

    # Exibir os estados, amplitudes e probabilidades
    probabilities = np.abs(psi) ** 2
    print("\nEstados, amplitudes e probabilidades:")
    for idx, state in enumerate(states):
        print(f"Estado '{state}': Amplitude = {psi[idx]}, Probabilidade = {probabilities[idx]:.4f}")

    idx_max = np.argmax(probabilities)
    recommended_state = states[idx_max]
    print(f"\nDecisão Recomendada: Estado '{recommended_state}' com probabilidade {probabilities[idx_max]:.4f}")

    # Otimização Evolutiva para ajustar multiplicadores dos pesos (opcional)
    use_evolution = input("\nDeseja executar otimização evolutiva para ajustar os pesos? (s/n): ").strip().lower()
    if use_evolution == 's':
        best_adjustments, best_fitness = genetic_optimization(psi, operators, base_weights, k_value, interactions,
                                                              uncertainties, num_simulations=500, num_generations=20,
                                                              pop_size=50)
        print(f"\nMelhor vetor de ajustes encontrado: {best_adjustments}")
        print(f"Melhor fitness (probabilidade favorável): {best_fitness * 100:.2f}%")
        # Aplicar os ajustes aos pesos base
        factor_list = list(operators.keys())
        raw_weights = np.array([base_weights[f] for f in factor_list])
        best_adjustments = np.array(best_adjustments)
        new_weights = raw_weights * best_adjustments
        new_weights /= new_weights.sum()
        adjusted_weights = {factor_list[i]: new_weights[i] for i in range(len(factor_list))}
        print("\nPesos ajustados após otimização evolutiva:")
        for factor in adjusted_weights:
            print(f"Fator '{factor}': Peso ajustado = {adjusted_weights[factor]:.4f}")
        # Reexecuta Monte Carlo com os novos pesos
        consensus, probability, simulation_results = monte_carlo_consensus_parallel(psi, operators, adjusted_weights,
                                                                                    k_value, interactions,
                                                                                    uncertainties, num_simulations)
        print(f"\nNovo Consenso M(A): {consensus:.4f}")
        print(f"Nova Probabilidade de decisão favorável (M(A) > 0): {probability * 100:.2f}%")

    # Visualizações Gráficas
    plt.figure(figsize=(10, 6))
    plt.hist(simulation_results, bins=30, color='violet', edgecolor='black')
    plt.title('Distribuição dos Valores de Mérito Moral Total M(A)')
    plt.xlabel('M(A)')
    plt.ylabel('Frequência')
    plt.grid(axis='y')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.bar(states, probabilities, color='skyblue')
    plt.title('Probabilidades dos Estados')
    plt.xlabel('Estados')
    plt.ylabel('Probabilidade')
    plt.grid(axis='y')
    plt.show()

    factor_list = list(operators.keys())
    exp_vals = [expected_values[f] for f in factor_list]
    uncert_vals = [uncertainties[f] for f in factor_list]
    x = np.arange(len(factor_list))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, exp_vals, width, label='Valor Esperado', color='green')
    ax.bar(x + width / 2, uncert_vals, width, label='Incerteza', color='orange')
    ax.set_ylabel('Valores')
    ax.set_title('Valores Esperados e Incertezas dos Fatores')
    ax.set_xticks(x)
    ax.set_xticklabels(factor_list)
    ax.legend()
    plt.grid(axis='y')
    plt.show()

    # Feedback Iterativo: permitir ajustes e nova simulação
    while True:
        feedback = input(
            "\nDeseja ajustar parâmetros (ex.: valor de k ou número de simulações) e reexecutar? (s/n): ").strip().lower()
        if feedback == 's':
            k_value = float(input("Digite o novo valor de k (nível de confiança): "))
            num_simulations = int(input("Digite o novo número de simulações Monte Carlo: "))
            consensus, probability, simulation_results = monte_carlo_consensus_parallel(psi, operators,
                                                                                        adjusted_weights, k_value,
                                                                                        interactions, uncertainties,
                                                                                        num_simulations)
            print(f"\nNovo Consenso M(A): {consensus:.4f}")
            print(f"Nova Probabilidade de decisão favorável (M(A) > 0): {probability * 100:.2f}%")
        else:
            break


if __name__ == "__main__":
    main()
