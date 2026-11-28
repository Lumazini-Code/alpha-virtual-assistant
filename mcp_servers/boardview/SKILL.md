---
name: boardview-repair
description: Use quando o usuário quiser diagnosticar, entender ou reparar uma placa eletrônica (notebook, celular, placa-mãe, fonte) tendo um boardview GenCAD (.cad) e o servidor MCP "boardview" disponível. Cobre rail sem tensão, curto ao GND, placa que não liga, caminho de alimentação, sequenciamento, localização física de componentes e cruzamento com medições feitas pelo técnico.
---

# Reparo de placas com o MCP `boardview`

O servidor lê um boardview GenCAD (`.cad`) e expõe **conectividade + geometria** da placa. Você atua como
assistente de bancada: raciocina sobre a topologia, diz **onde medir** e interpreta o que o técnico mede.

## O que o boardview sabe e o que NÃO sabe

Sabe: componentes (refdes, valor/código, pacote, lado TOP/BOTTOM, posição, rotação), pinos com coordenadas, rede de
cada pino, vias, testpoints, contorno.

NÃO sabe: esquemático, função de cada pino de um CI/FET, tensões esperadas, camadas internas, trilhas, se o
componente está bom. **Nunca afirme que um componente está defeituoso a partir do boardview.** Diga "suspeito",
indique como confirmar (medição) e o nível de confiança.

Tudo marcado como *heurística* (classe de componente, tipo de rede, tensão/estado inferidos do nome) pode estar
errado: trate como hipótese e confirme.

## Fluxo padrão

1. `load_board(path)` (se ainda não carregada) → `board_info()`: tamanho, contagens, GND, maiores CIs/conectores.
   `path` pode ser o `.cad`, a pasta que o contém (use `.` para a pasta de trabalho do servidor) ou um glob
   (`*.cad`); com vários candidatos a resposta lista os arquivos.
2. Oriente-se pelo problema:
   - **Alimentação / não liga / rail ausente** → `list_power_rails` → `power_rail_analysis(rail)` →
     `trace_connectivity` do rail até a fonte.
   - **Sinal / periférico** (SPI, I2C, USB, PCIe, clock) → `search_nets` por padrão do nome → `get_net`.
   - **Componente específico** → `get_part`; **o que há ao redor** → `neighbors`; **ver** → `render_region`.
3. Peça medições ao técnico (ver "Como pedir medições") e registre com `add_measurement`.
4. Rode `check_measurements_vs_board` para cruzar medições × topologia e atualizar o raciocínio.
5. Conclua com: hipótese principal, alternativas, próximo teste, risco.

Responda ao técnico com refdes e pinos concretos (ex.: "meça a tensão em `PL5` pino 2"), nunca só "verifique a
alimentação".

## Ferramentas (quando usar)

| Tool | Use para |
|---|---|
| `board_info` | primeira orientação; ICs grandes; quais redes são GND |
| `list_power_rails` | inventário de rails com tensão/estado inferidos do nome; filtre por `state` ('S5','S3','S0','AUX') ou `contains` |
| `power_rail_analysis(net)` | decoupling, fontes/série (L, FET, fusível), cargas, TVS e **ranking de suspeitos de curto** |
| `search_nets(pattern)` | achar redes por glob (`*PWRGD*`) ou regex (`re:VCC(IO|SA)`); mostra tags (i2c, usb, power_good, enable...) |
| `get_net(name)` | todos os componentes da rede; passivos de 2 pinos mostram a rede do outro lado |
| `get_part(refdes)` | valor decodificado (ex.: `SC10U6D3V…` = 10 µF/6,3 V), papel provável, pinos→redes. Em BGA use `pin_filter` |
| `search_parts(pattern)` | achar por refdes/valor/pacote; `part_class` filtra (mosfet, inductor, ic...) |
| `trace_connectivity(a,b)` | caminho por série (R/L/FB/fusível/0 Ω; FET/diodo marcados **ATIVO**). Segue rail da entrada ao consumidor |
| `neighbors(target)` | vizinhança física (inclui lado oposto) — dano por calor/líquido, capacitores sob BGA |
| `render_region(center,…)` | PNG da região; destaque redes/componentes. 10–40 mm de largura para ler refdes |
| `add_measurement` / `list_measurements` / `check_measurements_vs_board` / `clear_measurements` | memória de bancada e cruzamento |

Padrões de busca: glob case-insensitive; **sem curinga = "contém"**; prefixo `re:` para regex.
Alvos de `trace_connectivity`: refdes (`U17`), pino (`U17.5`) ou rede.

## Convenções de nomes (heurísticas, validar sempre)

- Tensão no nome: `3D3V` / `3P3V` / `3V3` = 3,3 V; `1D8V` = 1,8 V; `1V_VCCIO` = 1,0 V; `+19V` = 19 V.
- Estado de energia: `_S5` ≈ standby (presente com adaptador mesmo desligado); `_S3` suspensão; `_S0` só com a máquina
  ligada; `AUX`/`ALW` always-on; `DSW`/`RTC` poços do PCH/relógio.
- `DCBATOUT*` / `DC_IN*` = barramento adaptador/bateria (~7–20 V). O nome **não** é a tensão de saída dos ramos
  (`DCBATOUT_RT6575_5V` é entrada de um regulador de 5 V).
- Sufixos de controle: `_EN`, `_PG`/`PWRGD`/`POK`, `_FB`, `_PH`/`_SW`/`_LX` (nó de chaveamento), `_SS`, `_BOOT`.
  Aparecem como `signal`, não como rail. `#` = ativo em nível baixo.
- Prefixos variam por fabricante (ex.: `PC/PR/PL/PQ/PU` = parte de potência; `TC/PT` = capacitor polarizado;
  `RN` = rede de resistores; `PG` = gap/jumper de solda).
- Códigos de valor estilo Wistron/Quanta: `SC10U6D3V` = cap SMD 10 µF 6,3 V; `24D9R2F` = 24,9 Ω 1%;
  `3K3R6J` = 3,3 kΩ 5%; `0R…PAD` = jumper 0 Ω. O decodificador é *best-effort*.

## Playbooks

### A) Rail ausente (0 V ou baixo) 
1. `power_rail_analysis(rail)` → veja `possible_sources_or_series_elements` (indutor/FET/fusível) e cargas.
2. `trace_connectivity(entrada, rail)` (ex.: `DCBATOUT` → rail) para montar a cadeia: fonte → FET → indutor → rail.
3. Perguntar/medir em ordem: entrada do regulador → **EN**/**PG** do controlador (`search_nets '*_EN*'`) → nó de
   chaveamento (PH/SW) → saída. Rail ausente com entrada e EN presentes = controlador/FET/indutor/curto na saída.
4. Se a tensão cai ao ligar (sobe e some), suspeite de **curto/carga** na saída (proteção do regulador). Vá ao playbook B.
5. Se ausente só em `_S0`/`_S3`, confirme se o notebook realmente passou da fase anterior (verifique os `PG` e o `EN`
   que dependem dos rails de standby).

### B) Curto ao GND num rail
1. Peça resistência/diodo ao GND **com a placa desligada, sem bateria nem adaptador** (`add_measurement`, state
   `unpowered`). Valores de poucos ohms são normais em rails de CPU/núcleo; compare com um rail irmão.
2. `power_rail_analysis(rail)` → `short_to_ground_suspects_ranked`: capacitores primeiro, depois TVS/ESD, FETs, CIs.
3. Localize fisicamente com `render_region(center=<rail>, highlight_nets=[rail])` e `neighbors` (lembre dos
   componentes no **lado oposto**, sob BGAs).
4. Técnicas de confirmação a sugerir: injeção de tensão **limitada em corrente** no rail + câmera térmica/álcool;
   remover um suspeito por vez e remedir. Sempre avisar de riscos (ESD, calor, limitar corrente).

### C) Placa não liga / sem sinal de vida
1. `list_power_rails(state='S5')`: standby. Sem standby → problema na entrada (conector DC, fusível, FET de entrada,
   carregador). `trace_connectivity('<conector>', 'DCBATOUT')`.
2. Com standby OK: alimentação e clock do EC/KBC (`get_part(KBC…)`, `search_nets '*KBC*'`), botão de power,
   `RSMRST`/`PWRBTN`/`SLP_S*` (`search_nets 're:RSMRST|PWRBTN|SLP_S'`).
3. Depois: rails `_S3`/`_S0` e seus `PG`, então reset (`PLTRST`), clocks e o flash SPI (`search_nets '*SPI*'`).
4. Cada etapa só depende das anteriores: **encontre a primeira que falha**. Use as medições registradas para dizer
   até onde a placa chegou.

### D) Sinal/periférico (ex.: SPI, I2C)
`search_nets` pelo prefixo → `get_net` para ver origem, destino, resistores de série e pull-ups (resistor de um rail
até o sinal) → `get_part` do periférico. Para barramentos, compare a rede em todos os pontos de teste (`testpoints`).

## Como pedir medições

- Diga **ponto, modo e condição**: "tensão DC em `PL5.2` (modo V), com adaptador conectado, sem pressionar power".
- Resistência/diodo **sempre desligado** e descarregado; indique ponteira vermelha no ponto e preta no GND, e
  espere o valor estabilizar (rails com capacitores sobem lentamente).
- Peça poucas medições por vez, ordenadas pela maior capacidade de dividir o problema (bisseção da cadeia de energia).
- Registre cada resultado com `add_measurement` (inclua `state`). Medições em `GND` não têm significado.

## Como relatar

- Cite refdes, pinos, redes; diga se a informação vem de **dado do arquivo** (fato) ou **heurística** (nome/tipo).
- Dê hipótese principal + 1–2 alternativas + próxima medição + o que ela decide.
- Quando a topologia depender de função de pino que o boardview não traz (FET, CI), diga: "confirme no esquemático
  ou datasheet de `<valor do componente>`".
- Se faltar informação (modelo, estado de energia, o que já foi medido), pergunte **uma** coisa por vez.

## Armadilhas

- `trace_connectivity` assume condução por R/L/fusível/jumper; FET e diodo só conduzem conforme polarização/gate.
  Não conclua que há alimentação só porque há caminho.
- Redes enormes (GND, barramento) não são atravessadas pelo trace; relacione por componentes em série.
- Vista `BOTTOM` do `render_region` é **espelhada** (como virar a placa). Coordenadas em mm, Y para cima.
- 596 pinos sem rede em CIs grandes são normais (NC/pinos sem conexão no arquivo).
- Segurança: placa pode ter alta tensão do adaptador (≈19 V+) e energia em capacitores; desconectar bateria
  e adaptador antes de medir resistência/dessoldar; usar proteção ESD.
