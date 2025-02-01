import os
import numpy as np
import matplotlib.pyplot as plt
import openai

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')  # Make sure to set this environment variable securely

def normalize(vector):
    """
    Normaliza um vetor para que sua norma seja 1.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("A função de onda não pode ser o vetor nulo.")
    return vector / norm

def is_hermitian(matrix):
    """
    Verifica se uma matriz é Hermitiana, ou seja, se ela é igual à sua transposta conjugada.
    """
    return np.allclose(matrix, matrix.conj().T)

def make_hermitian(matrix):
    """
    Ajusta uma matriz para que seja Hermitiana, tomando a média entre a matriz e sua transposta conjugada.
    """
    return (matrix + matrix.conj().T) / 2

def get_operator_chatgpt(num_states, factor_name, states):
    """
    Usa a OpenAI ChatGPT API para gerar automaticamente o operador para um fator.

    Parâmetros:
    - num_states: número de estados de decisão.
    - factor_name: nome do fator.
    - states: lista com os nomes dos estados.

    Retorna:
    - operator: matriz Hermitiana representando o operador do fator.
    """
    print(f"\nGerando operador para o fator '{factor_name}' usando ChatGPT.")

    # Preparar o prompt para a API
    prompt = f"Para cada um dos seguintes estados de decisão, atribua um valor numérico de 1 a 10 para o fator '{factor_name}'.\n"
    for idx, state in enumerate(states):
        prompt += f"{idx + 1}. {state}\n"

    prompt += "\nForneça os valores no formato:\nEstado 1: valor1\nEstado 2: valor2\n..."

    try:
        # Chamar a API do OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use o modelo desejado
            messages=[
                {"role": "system", "content": "Você é um assistente que atribui valores numéricos a estados de decisão com base em fatores específicos."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )

        # Extrair a resposta
        chatgpt_reply = response.choices[0].message.content.strip()

        # Processar a resposta para extrair os valores
        factor_values = {}
        for line in chatgpt_reply.split('\n'):
            parts = line.strip().split(':')
            if len(parts) == 2:
                state_info, value = parts
                value = value.strip()
                # Extrair o número do estado
                state_num = int(''.join(filter(str.isdigit, state_info)))
                # Mapear para o nome do estado
                state_name = states[state_num - 1]
                factor_values[state_name] = float(value)
            else:
                continue  # Linha não reconhecida, pode ser ignorada

        # Construção do operador
        operator = np.zeros((num_states, num_states), dtype=complex)
        for i in range(num_states):
            state = states[i]
            value = factor_values.get(state, 5.0)  # Valor padrão se não encontrado
            operator[i, i] = value

        # Garantir que o operador seja Hermitiano
        if not is_hermitian(operator):
            operator = make_hermitian(operator)

        return operator

    except Exception as e:
        print(f"Erro ao gerar valores com ChatGPT: {e}")
        print("Usando valores padrão para o fator.")
        # Em caso de erro, usar valores padrão
        operator = np.identity(num_states) * 5.0
        return operator

def calculate_uncertainty(psi, operator):
    """
    Calcula o valor esperado e a incerteza (desvio padrão) de um operador dado o estado psi.

    Parâmetros:
    - psi: vetor de estado (função de onda normalizada).
    - operator: matriz representando o operador.

    Retorna:
    - expected_value: valor esperado do operador.
    - uncertainty: incerteza (desvio padrão) do operador.
    """
    # Valor esperado: <psi|operator|psi>
    expected_value = np.vdot(psi, operator @ psi).real
    # Valor esperado do quadrado do operador: <psi|operator^2|psi>
    expected_value_squared = np.vdot(psi, operator @ operator @ psi).real
    # Variância: sigma^2 = <operator^2> - <operator>^2
    variance = expected_value_squared - expected_value ** 2
    # Incerteza: sigma = sqrt(variância)
    uncertainty = np.sqrt(max(variance, 0))
    return expected_value, uncertainty

def main():
    print("Bem-vindo ao Algoritmo de Tomada de Decisão com Mérito Moral Total!\n")

    # Verificar se a API key está definida
    if not openai.api_key:
        print("Erro: A chave da API OpenAI não está definida. Por favor, defina a variável de ambiente 'OPENAI_API_KEY'.")
        return

    # Entrada dos estados de decisão
    num_states = int(input("Digite o número de estados de decisão: "))
    states = []
    for i in range(num_states):
        state_name = input(f"Digite o nome do estado {i+1}: ")
        states.append(state_name)

    # Entrada das amplitudes iniciais
    psi = np.zeros(num_states, dtype=complex)
    equal_amplitudes = input("\nDeseja usar amplitudes iguais para todos os estados? (s/n): ").strip().lower()
    if equal_amplitudes == 's':
        # Usa amplitudes iguais para reduzir viés
        psi[:] = 1
    else:
        # Solicita as amplitudes do usuário
        print("Digite as amplitudes iniciais (números complexos) para cada estado:")
        for idx, state in enumerate(states):
            amp = complex(input(f"Amplitude para o estado '{state}' (formato a+bj): "))
            psi[idx] = amp

    # Normalizar a função de onda
    psi = normalize(psi)

    # Definição dos operadores dos fatores
    num_factors = int(input("\nDigite o número de fatores a serem considerados: "))
    operators = {}          # Dicionário para armazenar os operadores
    expected_values = {}    # Dicionário para armazenar os valores esperados
    uncertainties = {}      # Dicionário para armazenar as incertezas

    for i in range(num_factors):
        factor_name = input(f"\nDigite o nome do fator {i+1}: ")
        operator = get_operator_chatgpt(num_states, factor_name, states)
        # Calcular o valor esperado e a incerteza para o fator
        expected_value, uncertainty = calculate_uncertainty(psi, operator)
        operators[factor_name] = operator
        expected_values[factor_name] = expected_value
        uncertainties[factor_name] = uncertainty

    # Cálculo dos pesos inversamente proporcionais às variâncias
    variances = {k: uncertainties[k]**2 for k in uncertainties}
    total_inverse_variance = sum([1/v if v > 0 else 0 for v in variances.values()])
    weights = {}
    for k in variances:
        if variances[k] > 0:
            weights[k] = (1 / variances[k]) / total_inverse_variance
        else:
            weights[k] = 0  # Se a variância é zero, o peso é zero
    # Normalizar os pesos para que a soma seja 1
    weight_sum = sum(weights.values())
    if weight_sum == 0:
        raise ValueError("A soma dos pesos é zero. Verifique as variâncias dos fatores.")
    for k in weights:
        weights[k] /= weight_sum

    # Exibir os pesos calculados
    print("\nPesos dos fatores (calculados inversamente às variâncias):")
    for k in weights:
        print(f"Fator '{k}': Peso = {weights[k]:.4f}")

    # Valor de k (nível de confiança)
    k_value = float(input("\nDigite o valor da constante k (nível de confiança, ex.: 1 para 68%, 2 para 95%): "))

    # Definição das interações entre fatores
    interactions = {}
    response = input("\nDeseja definir interações entre os fatores? (s/n): ").strip().lower()
    if response == 's':
        num_interactions = int(input("Digite o número de interações: "))
        for _ in range(num_interactions):
            factor_i = input("Digite o nome do primeiro fator da interação: ")
            factor_j = input("Digite o nome do segundo fator da interação: ")
            theta = float(input(f"Digite o coeficiente de interação (theta) entre '{factor_i}' e '{factor_j}': "))
            # Calcular o valor esperado do produto dos operadores: <psi|F_i F_j|psi>
            product_operator = operators[factor_i] @ operators[factor_j]
            expected_product = np.vdot(psi, product_operator @ psi).real
            interactions[(factor_i, factor_j)] = {
                'theta': theta,
                'expected_product': expected_product
            }
    else:
        print("Nenhuma interação definida.")

    # Cálculo do Mérito Moral Total M(A)
    M_A = 0.0

    # Primeiro termo: Soma ponderada dos valores esperados
    first_term = sum([weights[factor] * expected_values[factor] for factor in operators])
    M_A += first_term

    # Segundo termo: Subtração das incertezas ponderadas
    second_term = k_value * sum([weights[factor] * uncertainties[factor] for factor in operators])
    M_A -= second_term

    # Terceiro termo: Soma das interações
    third_term = 0.0
    for interaction in interactions.values():
        third_term += interaction['theta'] * interaction['expected_product']
    M_A += third_term

    # Exibição dos resultados
    print(f"\nO Mérito Moral Total (M(A)) é: {M_A:.4f}")
    print("\nEstados, amplitudes e probabilidades:")
    probabilities = np.abs(psi)**2
    for idx, state in enumerate(states):
        amp = psi[idx]
        prob = probabilities[idx]
        print(f"Estado '{state}': Amplitude = {amp}, Probabilidade = {prob:.4f}")

    # Decisão recomendada com base nas probabilidades
    idx_max = np.argmax(probabilities)
    recommended_state = states[idx_max]
    print(f"\nDecisão Recomendada: Estado '{recommended_state}' com probabilidade {probabilities[idx_max]:.4f}")

    # Visualizações gráficas

    # Gráfico de probabilidades dos estados
    plt.figure(figsize=(8, 6))
    plt.bar(states, probabilities, color='skyblue')
    plt.title('Probabilidades dos Estados')
    plt.xlabel('Estados')
    plt.ylabel('Probabilidade')
    plt.grid(axis='y')
    plt.show()

    # Gráfico de valores esperados e incertezas
    factors = list(operators.keys())
    exp_vals = [expected_values[factor] for factor in factors]
    uncertainties_vals = [uncertainties[factor] for factor in factors]

    x = np.arange(len(factors))
    width = 0.35  # Largura das barras

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, exp_vals, width, label='Valor Esperado', color='green')
    rects2 = ax.bar(x + width/2, uncertainties_vals, width, label='Incerteza', color='orange')

    ax.set_ylabel('Valores')
    ax.set_title('Valores Esperados e Incertezas dos Fatores')
    ax.set_xticks(x)
    ax.set_xticklabels(factors)
    ax.legend()
    plt.grid(axis='y')
    plt.show()

if __name__ == "__main__":
    main()
