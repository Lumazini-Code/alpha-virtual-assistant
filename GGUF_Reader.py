import logging
import sys
from pathlib import Path
from gguf.gguf_reader import GGUFReader
import os

def buscar_arquivo(nome_arquivo, pasta_raiz):
    for raiz, _, arquivos in os.walk(pasta_raiz):
        if nome_arquivo in arquivos:
            pathModel = os.path.join(raiz, nome_arquivo)
            return True, pathModel 
    return False

dirname = fR"C:\Users\{os.getlogin()}\.cache\huggingface\hub"

#!/usr/bin/env python3

logger = logging.getLogger("reader")

# Necessary to load the local gguf package
sys.path.insert(0, str(Path(__file__).parent.parent))




def get_num_layers_from_tensors(reader):
    """
    Infere o número de layers baseado nos nomes dos tensores.
    Exemplo esperado: blk.0.attn_q.weight
    """
    layer_indices = set()

    for tensor in reader.tensors:
        name = tensor.name

        if name.startswith("blk."):
            try:
                idx = int(name.split(".")[1])
                layer_indices.add(idx)
            except Exception:
                continue

    if layer_indices:
        return max(layer_indices) + 1  # começa do 0

    return None


def read_gguf_file(gguf_file_path):
    """
    Reads and prints key-value pairs and tensor information from a GGUF file in an improved format.
    """

    reader = GGUFReader(gguf_file_path)

    # -------------------------
    # KEY-VALUE METADATA
    # -------------------------
    print("Key-Value Pairs:")
    max_key_length = max(len(key) for key in reader.fields.keys())

    for key, field in reader.fields.items():
        value = field.parts[field.data[0]]
        print(f"{key:{max_key_length}} : {value}")

    print("----")

    # -------------------------
    # LAYERS (INFERIDO)
    # -------------------------
    num_layers = get_num_layers_from_tensors(reader)

    if num_layers is not None:
        print(f"Total de layers (inferido pelos tensores): {num_layers}")
    else:
        print("Não foi possível inferir o número de layers.")
    return num_layers

if __name__ == '__main__':
    if len(sys.argv) < 2:
        logger.info("Usage: reader.py <path_to_gguf_file>")
        sys.exit(1)

    gguf_file = sys.argv[1]
    _, gguf_file_path = buscar_arquivo(gguf_file, dirname)
    layers = read_gguf_file(gguf_file_path)

    with open(f"metadata\\{gguf_file}.metadata", "w") as f:
        f.write(str(layers))