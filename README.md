
# Logic Decision Engine (LDG) - Experimental

**Logic Decision Engine (LDG)** é um sistema híbrido de apoio à decisão que integra fundamentos éticos e morais com técnicas avançadas de simulação e otimização. Este projeto é um modelo experimental e visa explorar a personalização e automação na tomada de decisões com base em perfis morais, utilizando dados simulados e técnicas de aprendizado híbrido.

## Visão Geral

O LDG utiliza:
- **Questionário Moral**: Um breve questionário inspirado no *Moral Foundations Questionnaire (MFQ)* para capturar o perfil ético/moral do usuário.
- **Geração Automática de Operadores**: Operadores de decisão são gerados automaticamente com base nos nomes dos fatores. São utilizados valores padrão a partir de um banco de dados interno e, quando possível, a API do ChatGPT.
- **Cálculo Automático de Incertezas**: Estima valores esperados e desvios padrão (incertezas) dos operadores com base na função de onda.
- **Simulação Monte Carlo Paralela**: Executa simulações paralelas para obter uma distribuição dos resultados e calcular um consenso final, juntamente com a probabilidade de a decisão ser favorável.
- **Otimização Evolutiva (Opcional)**: Algoritmos evolutivos são utilizados para ajustar os pesos dos fatores, maximizando a probabilidade de um resultado positivo.
- **Feedback Iterativo**: Permite que o usuário revise os resultados e ajuste parâmetros (como o nível de confiança e número de simulações) para refinar a decisão.

## Principais Funcionalidades

- **Entrada Mínima**: O usuário insere apenas os nomes dos estados (por exemplo, "Dormir Agora, Dormir Mais Tarde") e dos fatores (por exemplo, "Descanso, Produtividade, Saúde, BemEstarSocial").
- **Perfil Moral Personalizado**: Um questionário baseado no MFQ coleta dados para ajustar dinamicamente os pesos dos fatores, refletindo a visão ética do usuário.
- **Operadores Automatizados**: Valores dos operadores são gerados automaticamente (via API ou banco de dados) e validados para garantir que sejam Hermitianos.
- **Simulação e Otimização**: Utiliza simulação Monte Carlo paralela e, opcionalmente, algoritmos evolutivos para refinar os parâmetros e chegar a um consenso final.
- **Visualizações Gráficas**: Gera gráficos que mostram a distribuição dos valores de \( M(A) \), probabilidades dos estados e os valores esperados/incertezas dos fatores.

## Aviso Experimental

**Este é um modelo experimental.**  
Os resultados e parâmetros gerados por este sistema são baseados em simulações e modelos iniciais, e o sistema não deve ser usado para decisões críticas sem uma validação adicional. Contribuições e feedback são muito bem-vindos para aprimorar e validar este modelo.

## Instalação

### Requisitos

- Python 3.6+
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [OpenAI Python API](https://github.com/openai/openai-python) (opcional, para geração automática dos operadores)
- [DEAP](https://github.com/DEAP/deap) (opcional, para otimização evolutiva)

### Instalação via pip

```bash
pip install numpy matplotlib openai deap
```

### Configuração da API do OpenAI ou DeepSeek (opcional)

Defina sua chave de API do OpenAI na variável de ambiente `OPENAI_API_KEY`:

- **Unix/Linux/MacOS:**

  ```bash
  export OPENAI_API_KEY="sua-chave-api-aqui"
  ```

- **Windows:**

  ```bash
  setx OPENAI_API_KEY "sua-chave-api-aqui"
  ```

## Uso

1. **Execute o Programa:**

   ```bash
   python merito_moral_automatizado.py
   ```

2. **Forneça os Dados Solicitados:**
   - Insira os estados de decisão (ex.: "Dormir Agora, Dormir Mais Tarde").
   - Insira os fatores (ex.: "Descanso, Produtividade, Saúde, BemEstarSocial").
   - Responda ao questionário moral para determinar seu perfil ético.
   - O sistema gerará automaticamente os operadores para cada fator, calculará os valores esperados e as incertezas, ajustará os pesos conforme seu perfil, e executará simulações Monte Carlo em paralelo para obter o consenso final e a probabilidade da decisão.
   - (Opcional) Execute a otimização evolutiva para refinar os ajustes dos pesos.
   - Utilize o feedback iterativo para ajustar os parâmetros e reexecutar a simulação, se desejado.

3. **Visualize os Resultados:**
   - O sistema exibirá o valor do Mérito Moral Total \( M(A) \), a distribuição dos resultados e gráficos ilustrativos.

## Contribuições

Contribuições são bem-vindas! Se você deseja melhorar o projeto, por favor, sinta-se à vontade para enviar um *pull request* ou abrir uma *issue* com sugestões, melhorias ou correções.

## License

Este projeto está licenciado sob a [MIT License](LICENSE).

## Contato

Para dúvidas, sugestões ou contribuições, entre em contato através do GitHub.


---
A seguir, apresento uma simulação de teste para a decisão: "Devo mudar de casa (para ter um quarto extra) ou reformar a casa atual para construir um novo quarto?" Usarei o modelo híbrido (com paralelismo, ajuste dinâmico dos pesos via questionário moral, simulação Monte Carlo e, opcionalmente, otimização evolutiva) que descrevemos anteriormente. Note que este exemplo é **experimental** e utiliza dados simulados e valores padrão.

---

## Simulação de Teste

### 1. **Definição do Cenário**

**Estados de Decisão:**  
- **Estado 1:** Mudar de casa  
- **Estado 2:** Reformar a casa atual

**Fatores Considerados:**  
Para este exemplo, vamos considerar três fatores que influenciam a decisão:

1. **Custo:**  
   - *Mudar de casa* normalmente envolve um custo mais alto (despesas de mudança, aquisição de novo imóvel, etc.).  
   - *Reformar a casa* pode ter um custo menor, pois envolve apenas a reforma e ampliação.

2. **Conforto:**  
   - Um imóvel novo (opção de mudar) pode oferecer um ambiente mais confortável, moderno ou adaptado às necessidades.  
   - Reformar pode melhorar o conforto, mas geralmente tem limitações quanto à adaptação.

3. **Localização:**  
   - Mudar de casa pode oferecer a oportunidade de escolher uma localização melhor (acesso a serviços, segurança, etc.).  
   - Reformar a casa atual mantém a mesma localização, com suas vantagens e desvantagens.

### 2. **Valores Padrão e Dados Utilizados (Exemplo Simulado)**

Para cada fator, o sistema gera operadores automaticamente (neste exemplo, matrizes diagonais) com valores padrão definidos em um banco de dados interno:

- **Custo:**  
  - *Mudar de casa:* 6 (valor mais alto, indicando maior custo)  
  - *Reformar a casa:* 4 (menor custo)  

- **Conforto:**  
  - *Mudar de casa:* 8 (imóvel novo com maior conforto)  
  - *Reformar a casa:* 6 (melhoria, mas não tão alta)

- **Localização:**  
  - *Mudar de casa:* 9 (possibilidade de escolher uma localização melhor)  
  - *Reformar a casa:* 5 (mantém a localização atual)

Os operadores serão então representados (para cada fator) como matrizes diagonais. Por exemplo, para o fator **Custo**:

\[
\hat{F}_{\text{Custo}} = \begin{pmatrix} 6 & 0 \\ 0 & 4 \end{pmatrix}
\]

E o vetor de estado \( \psi \) (representando a probabilidade inicial de cada alternativa) é definido automaticamente como:
\[
|\Psi\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix}1 \\ 1\end{pmatrix}
\]
(se as amplitudes forem iguais, a probabilidade de cada estado será 50% se não houver diferença).

### 3. **Ajuste dos Pesos via Perfil Moral**

Antes da entrada dos fatores, o sistema aplica um questionário moral baseado no MFQ. Por exemplo, o usuário responde (em uma escala de 1 a 10):

- **Harm (evitar danos):** 8  
- **Fairness (justiça):** 7  
- **Loyalty (lealdade):** 5  
- **Authority (autoridade):** 4  
- **Purity (pureza):** 6  

Com base em um mapeamento pré-definido (por exemplo, “Custo” pode ser sensível à justiça, “Conforto” à proteção contra danos, e “Localização” à pureza ou autoridade), os pesos ajustados podem ser, por exemplo:

- **Custo:** Peso ajustado = 0.35  
- **Conforto:** Peso ajustado = 0.40  
- **Localização:** Peso ajustado = 0.25

Esses pesos fazem com que, ao combinar os valores dos fatores, as diferenças entre as alternativas se tornem mais evidentes.

### 4. **Simulação Monte Carlo Paralela**

Em seguida, o modelo executa simulações Monte Carlo em paralelo. Em cada simulação:

- Para cada fator, o sistema amostra um valor a partir de uma distribuição normal com:
  - **Média:** o valor esperado do fator, como \(\langle \Psi | \hat{F}_i | \Psi \rangle\)
  - **Desvio padrão:** a incerteza \( \sigma_{F_i} \)
  
- A fórmula utilizada para calcular \( M(A) \) em cada simulação é:

\[
M(A) = \sum_{i} w_i \langle \Psi | \hat{F}_i | \Psi \rangle - k \cdot \sum_{i} w_i \sigma_{F_i}
\]

*Nota: Aqui ignoramos, para simplificação, termos de interação (que podem ser incluídos se houver interações definidas).*

- Suponha que \( k = 1 \).

Após 1.000 simulações, o sistema calcula:
- A **média** dos valores de \( M(A) \) (consenso).
- A **proporção** das simulações em que \( M(A) > 0 \) (indicando a probabilidade de que a decisão seja favorável).

### 5. **Exemplo de Resultado Simulado**

Suponha que os cálculos tenham gerado os seguintes resultados:

- **Média de \( M(A) \) (consenso):** 6.5  
- **Probabilidade de \( M(A) > 0 \):** 70%

Esse resultado indicaria que, com base nos dados e no perfil moral do usuário, a opção de **"Mudar de casa"** é favorecida (consenso positivo) e, nas simulações, 70% dos casos resultaram em um valor de \( M(A) \) positivo.

### 6. **Interpretação dos Resultados**

- **Consenso \( M(A) = 6.5 \):**  
  Um valor positivo e razoavelmente alto indica que os fatores, quando ponderados e ajustados conforme o perfil moral, sugerem que mudar de casa (para ter um quarto extra) é a alternativa mais vantajosa.

- **Probabilidade de Decisão Favorável = 70%:**  
  Isso significa que, em 70% das simulações, o modelo considerou a opção "Mudar de casa" como preferível.

- **Razões para Resultados Diferenciados:**  
  Se os valores inseridos ou gerados automaticamente não apresentarem assimetria significativa entre as alternativas, o modelo pode tender a dar um resultado próximo de 50% (probabilidade igual para ambas). Neste exemplo, assumimos que os dados (custos, conforto, localização) são diferenciados o suficiente para gerar um consenso favorável a uma alternativa.

### 7. **Conclusão**

Com base nessa simulação:
- **Decisão Recomendada:** Mudar de casa para ter um quarto extra, pois o modelo mostra um consenso \( M(A) \) positivo (6.5) e uma probabilidade de 70% de ser a decisão mais favorável.
- **Observações:**  
  Se o modelo sempre apresentar resultados próximos de 50%, isso pode indicar que os valores dos fatores (ou as amplitudes) estão muito simétricos. Para obter resultados mais diferenciados, é fundamental que os dados (custos, conforto, localização) reflitam diferenças reais entre as alternativas.

---

Esta simulação exemplifica como o modelo híbrido pode ser aplicado para tomar decisões complexas, utilizando múltiplas simulações para gerar um consenso e uma probabilidade final da decisão, e como o perfil moral do usuário pode influenciar os ajustes dos pesos. Se precisar de mais ajustes ou de mais detalhes sobre a implementação, estou à disposição para ajudar!
---

Esta descrição fornece uma visão abrangente do projeto, destaca suas funcionalidades e deixa claro que o modelo é experimental. Se precisar de mais alguma modificação ou ajuste, estou à disposição para ajudar!
