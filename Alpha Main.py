import speech_recognition as sr
from PySimpleGUI import PySimpleGUI as sg
import pyttsx3
import datetime
import wikipedia
import pywhatkit
import random
import os
import webbrowser
import sys
import requests
import bs4
import pygame
import time

def pyaudio(caminho):
    pygame.init()
    pygame.mixer.music.load(caminho)
    pygame.mixer.music.play()
    pygame.event.wait()
    time.sleep(1)


with open('Responce.dll', "w") as respiosta:
    slaaaaaaaaa = respiosta.write('')

with open('username.dll', "r") as name:
    while True:
        username = name.read()
        break

audio = sr.Recognizer()
maquina = pyttsx3.init()





def iafala(comando):
    with open('Responce.dll', "w") as respiosta:
        slaaaaaaaaa = respiosta.write(comando)
    engine = pyttsx3.init()
    engine.say(comando)
    engine.runAndWait()




def executa_comando():

    try:
        with sr.Microphone() as source:
            pyaudio('Sounds\Audio init.mp3')
            voz = audio.listen(source)
            comando = audio.recognize_google(voz, language='pt-BR')
            comando = comando.lower()
            comando2 = audio.recognize_google(voz, language='pt-BR')
            comando2 = comando.lower()
            pyaudio('Sounds\Audio disable.mp3')
            
            

            if 'alfa' in comando:
                comando = comando.replace('alfa', '')
                maquina.runAndWait()
            print(comando)

        with open('ask.dll',"w") as aubleu:
            resposti = aubleu.write(comando2)

    except:
        iafala(username + ', eu não consegui entender oque você disse. verifique o seu microfone ou diminua os sons externos')
        
    return comando





def comando_voz_usuario():
    with open('Responce.dll', "w") as respiosta:
        slaaaaaaaaa = respiosta.write('')

    with open('ask.dll', "w") as iasgfo:
        sojfp = iasgfo.write('')



    comando = executa_comando()


    if 'horas' in comando:
        horas = comando.replace('horas', '')
        hora = datetime.datetime.now().strftime('%H:%M')
        iafala(username + ', agora são' + hora)

    elif 'horario' in comando:
        horario = comando.replace('horario', '')
        hora = datetime.datetime.now().strftime('%H:%M')
        iafala(username + ', agora são' + hora)

    if 'toque' in comando:
        musica = comando.replace('toque','')
        pywhatkit.playonyt(musica)
        iafala(random.choice(listatocar) + musica)

    elif 'tocar' in comando:
        tocar = comando.replace('tocar','')
        pywhatkit.playonyt(tocar)
        iafala(random.choice(listatocar) + tocar)


    if 'apresentar' in comando:
        iafala(random.choice(listapresentar))

    elif 'apresente se' in comando:
        iafala(random.choice(listapresentar))

    if 'se apresente' in comando:
        iafala(random.choice(listapresentar))



    if 'curiosidade' in comando:
        iafala(username + ', você sabia que' + random.choice(curiosidade))

    elif 'curiosidades' in comando:
        iafala(username + ', você sabia que' + random.choice(curiosidade))


    if 'piada' in comando:
        iafala(random.choice(prepiada) + random.choice(piadas))
        pyaudio(random.choice(pospiada))

    elif 'piadas' in comando:
        iafala(random.choice(prepiada) + random.choice(piadas))
        pyaudio(random.choice(pospiada))
        

    if 'dia' and 'hoje' in comando:
        hora = datetime.datetime.now().strftime(username + ', hoje é dia' + '%d' + 'de' + '%m' + 'de' + '%Y')
        iafala(hora)

    elif 'navegador' in comando:
        nave = comando.replace('navegador', '')
        webbrowser.open('https://www.google.com')
        iafala('ok ' + username + ', Abrindo Navegador')
        

    if 'steam' in comando:
        steam = comando.replace('steam', '')
        os.startfile("C:\Program Files (x86)\Steam\Steam.exe")
        iafala('ok ' + username + ', abrindo istim')

    elif 'por' and 'você' and 'criada' in comando:
        iafala(username + ' Eu fui criada em 2023 pelo felipe lumazini')


    if 'sair' in comando:
        iafala('ok' + username + ', saindo')
        sys.exit()

    elif 'fechar' in comando:
        iafala('ok' + username + ', fechando')
        sys.exit()

    if 'parar' in comando:
        iafala('ok ' + username + ', parando')
        sys.exit()

    if 'desligar' in comando:
        os.startfile("Shutdown lnk.lnk")
        iafala(username + ', seu computador será desligado em 10 segundos, salve suas coisas.')
        sys.exit()


    



    if 'youtube' in comando:
        webbrowser.open('https://www.youtube.com')
        iafala('ok ' + username + ', abrindo youtube')

    

    
    def get_crypto_price(coin):
        url = "https://www.google.com/search?q=" + coin + " hoje"
        HTML = requests.get(url)
        soup = bs4.BeautifulSoup(HTML.text, 'html.parser')
        text = soup.find("div", attrs={'class': 'BNeawe iBp4i AP7Wnd'}).find("div",
                                                                            attrs={
                                                                                'class': 'BNeawe iBp4i AP7Wnd'}).text
        iafala( username + f' O preço de {coin} é de {text}')



    if 'valor hoje' in comando:
        coin = comando.replace('valor hoje', '')
        get_crypto_price(coin)


    elif 'jogue' and 'moeda' in comando:
        moeda = ['cara', 'coroa']
        iafala(username + ', o resultado foi. ' + random.choice(moeda))           

    if 'obrigado' in comando:
        # Listas
        denada = ['De nada, ', 'Por nada, ', 'A seu dispor! ', 'Até logo!']
        denada2 = random.choice(denada)
        iafala(denada2 + username )


    elif 'legal' in comando:
        obrigado = ['fico muito feliz que tenha gostado',
                    ]
        iafala(random.choice(obrigado))

    if 'navegar' in comando or 'navegue' in comando:
        navegar = comando.replace('navegar para', '')
        navegar = comando.replace('navegue para', '')
        webbrowser.open(f'https://www.google.com.br/maps/search/{navegar}/')
        iafala(f'trasando róta para {navegar}')

    if 'trace uma rota' in comando:
        navegar2 = comando.replace('trace uma rota', '')
        navegar2 = comando.replace('para', '')
        webbrowser.open(f'https://www.google.com.br/maps/search/{navegar2}/')
        iafala(f'trasando róta para {navegar2}')

    elif 'erros' in comando:
        os.startfile('Bugscan lnk.lnk')
        iafala('um aplicativo com o nome de Windows Repair By Lumazini Play foi iniciado. siga as instruções dadas pelo aplicativo para menos confusão')

    if 'história' in comando:
        iafala(random.choice(prehist) + random.choice(historia))

    elif 'repitir' in comando:
        repitir = comando.replace('repitir', '')
        iafala(repitir)

    elif 'repita' in comando:
        repitir = comando.replace('repita', '')
        iafala(repitir)

    if 'você' in comando and 'vivo' in comando:
        iafala(username + ', eu, como uma inteligência artificial, não estou realmente viva, porém não estou morta, é meio complicado explicar, mais, não, não estou viva.')


    elif 'Qual' and 'seu' and 'nome' in comando:
        iafala(f'{username}, meu nome é alfa, prazer em conhecer-lo')

    if 'Como' and 'você' and 'está' in comando:
        iafala(f'{username}, eu sendo uma inteligência artificial, não tenho sentimentos para saber se estou bem ou mal, mais obrigado por perguntar')

    elif 'O que' and 'você' and 'faz' in comando:
        iafala(f'{username}, eu posso responder perguntas. contar piadas. contar histórias e muito mais!')

    if 'quem' in comando  and 'criou' in comando and 'você' in comando:
        iafala(f'{username}, eu criada por um programador de python em 2023.')

    if 'quem é' in comando:
        procurar = comando.replace('quem é', '')
        wikipedia.set_lang('pt')
        resultado = wikipedia.summary(procurar,2)
        iafala(username + ', ' + resultado)
        iafala(f'aqui estão algumas imagems sobre. {procurar}')
        webbrowser.open(f'https://www.google.com/search?q={procurar}&hs=NXX&hl=pt-BR&sxsrf=AJOqlzXhn_IkrrzxjUWx6-wIFbwO5HPiKg:1679610329333&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjDuKmIjPP9AhVTrZUCHaSDANEQ_AUoAXoECAEQAw&biw=1495&bih=754&dpr=1.25')


    if 'o que é' in comando:
        procurar = comando.replace('o que é', '')
        wikipedia.set_lang('pt')
        resultado = wikipedia.summary(procurar,4)
        print(resultado)
        iafala(username + ', '  + resultado)
        iafala(f'aqui estão algumas imagems sobre. {procurar}')
        webbrowser.open(f'https://www.google.com/search?q={procurar}&hs=NXX&hl=pt-BR&sxsrf=AJOqlzXhn_IkrrzxjUWx6-wIFbwO5HPiKg:1679610329333&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjDuKmIjPP9AhVTrZUCHaSDANEQ_AUoAXoECAEQAw&biw=1495&bih=754&dpr=1.25')

    if 'quem foi' in comando:
        procurar = comando.replace('quem é', '')
        wikipedia.set_lang('pt')
        resultado = wikipedia.summary(procurar,2)
        iafala(username + ', '  + resultado)
        iafala(f'aqui estão algumas imagems sobre. {procurar}')
        webbrowser.open(f'https://www.google.com/search?q={procurar}&hs=NXX&hl=pt-BR&sxsrf=AJOqlzXhn_IkrrzxjUWx6-wIFbwO5HPiKg:1679610329333&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjDuKmIjPP9AhVTrZUCHaSDANEQ_AUoAXoECAEQAw&biw=1495&bih=754&dpr=1.25')


    if 'o que foi' in comando:
        procurar = comando.replace('o que é', '')
        wikipedia.set_lang('pt')
        resultado = wikipedia.summary(procurar,4)
        print(resultado)
        iafala(username + ', '  + resultado)
        iafala(f'aqui estão algumas imagems sobre. {procurar}')
        webbrowser.open(f'https://www.google.com/search?q={procurar}&hs=NXX&hl=pt-BR&sxsrf=AJOqlzXhn_IkrrzxjUWx6-wIFbwO5HPiKg:1679610329333&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjDuKmIjPP9AhVTrZUCHaSDANEQ_AUoAXoECAEQAw&biw=1495&bih=754&dpr=1.25')



    
    if 'qual' in comando:
        procurar = comando.replace('qual', '')
        wikipedia.set_lang('pt')
        resultado = wikipedia.summary(procurar,2)
        iafala(username + ', '  + resultado) 
        iafala(f'aqui estão algumas imagems sobre. {procurar}')
        webbrowser.open(f'https://www.google.com/search?q={procurar}&hs=NXX&hl=pt-BR&sxsrf=AJOqlzXhn_IkrrzxjUWx6-wIFbwO5HPiKg:1679610329333&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjDuKmIjPP9AhVTrZUCHaSDANEQ_AUoAXoECAEQAw&biw=1495&bih=754&dpr=1.25')

    if 'explique' in comando:
        procurar = comando.replace('explique', '')
        wikipedia.set_lang('pt')
        resultado = wikipedia.summary(procurar,2)
        iafala(username + ', '  + resultado) 
        iafala(f'aqui estão algumas imagems sobre. {procurar}')
        webbrowser.open(f'https://www.google.com/search?q={procurar}&hs=NXX&hl=pt-BR&sxsrf=AJOqlzXhn_IkrrzxjUWx6-wIFbwO5HPiKg:1679610329333&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjDuKmIjPP9AhVTrZUCHaSDANEQ_AUoAXoECAEQAw&biw=1495&bih=754&dpr=1.25')

    elif 'explicar' in comando:
        procurar = comando.replace('explicar', '')
        wikipedia.set_lang('pt')
        resultado = wikipedia.summary(procurar,2)
        iafala(username + ', '  + resultado) 
        iafala(f'aqui estão algumas imagems sobre. {procurar}')
        webbrowser.open(f'https://www.google.com/search?q={procurar}&hs=NXX&hl=pt-BR&sxsrf=AJOqlzXhn_IkrrzxjUWx6-wIFbwO5HPiKg:1679610329333&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjDuKmIjPP9AhVTrZUCHaSDANEQ_AUoAXoECAEQAw&biw=1495&bih=754&dpr=1.25')

    if 'teste' in comando and 'internet' in comando:
        os.startfile('Internet speed test.py')

    elif 'imagem' in comando:
        imagem = comando.replace('imagem', '')
        imagem = comando.replace('busque', '')
        imagem = comando.replace('uma', '')
        imagem = comando.replace('sobre', '')
        iafala(f'aqui estão algumas imagems sobre: {imagem}')
        webbrowser.open(f'https://www.google.com/search?q={imagem}&hs=NXX&hl=pt-BR&sxsrf=AJOqlzXhn_IkrrzxjUWx6-wIFbwO5HPiKg:1679610329333&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjDuKmIjPP9AhVTrZUCHaSDANEQ_AUoAXoECAEQAw&biw=1495&bih=754&dpr=1.25')





piadas = [
    "Por que a aranha é o animal mais carente do mundo? Porque ela é um arac'needyou'.",
    "Por que o pinheiro não se perde na floresta? Porque ele tem uma pinha.",
    "Sabe como é a piada do pintinho caipira? Pir.",
    "Um caipira chega na casa de um amigo que está vendo TV e pergunta: E aí, firme? O outro responde: Não, futebor",
    "O que o pagodeiro foi fazer na igreja? Foi canta 'pá god'.",
    "Por que Napoleão era chamado sempre para as festas na França? Porque ele era 'Bom Na Party'.",
    "O que aconteceu com os lápis quando souberam que o dono da Faber Castell morreu? Ficaram desapontados",
    "A plantinha foi ao hospital, mas não foi atendida. Por quê? Porque lá só tinha médico de 'plantão'.",
    "Você conhece o site do cavalo? O www ponto cavalo ponto com ponto com ponto com ponto com ponto com'.",
    "Você conhece a piadinha do ponêi? Pô nei eu",
    "Qual é a fórmula da água benta? H Deus O.",
    "Qual é o rei dos queijos? rei queijão.",
    "O que o pato falou para a pata? Vem quá",
    "Por que a velhinha não usa relógio? Porque ela é uma sem hora.",
    "O que a vaca disse para o boi? Te amuuuuuuu.",
    "O que que Xuxa foi fazer no bar? Beber ca Sasha.",
    "havia dois caminhões voando. Um caiu. Por que o outro continuou voando? Porque era caminhão-pipa.",
    "Por que a formiga tem quatro patas? Porque se tivesse cinco patas se chamaria fivemiga",
    "Quando os americanos comeram carne pela primeira vez? Quando chegou Cristóvão Com Lombo",
    "o funcionário fala para o seu chefe. Chefe, quero um aumento. Saiba o senhor que tem três empresas atrás de mim. e o chefe assustado responde. Quais? e o funcionário diz. A de água, a de luz e a de telefone.",
    "Sabe o que o melão estava fazendo de mãos dadas com o mamão perto de Copacabana? Levando o mamão papaia.",
    "Esse salgado é de hoje? Não, é de ontem. E como faço pra comer o de hoje? Volte amanhã!",
    "Me vê dois ingressos, por favor. É para Romeu e Julieta? Não, é para mim e para minha namorada mesmo.",
    "Qual é a panela que está sempre triste? A panela depressão.",
    "Você gosta de bonecas russas? Eu não, elas são muito cheias de si.",
    "Qual é o oposto de volátil? Vem cá sobrinho.",
    "Por que as plantinhas não falam? Porque elas são mudinhas.",
    "Qual é a fórmula da água benta? H Deus O!",
    "Por que o jogador de golfe comprou calças novas? Porque tinha um buraco.",
    "O que os estilistas fazem no tempo livre? Inventam moda.",
    "Qual a cidade brasileira onde não tem táxis? Uberlândia.",
    "Qual é o peixe que caiu do 10º andar? O aaaaaaatum.",
    "Como se chama alguém que nasceu no Brasil, viveu na Escócia, se mudou para a África e morreu na China? Defunto.",
    "Quer saber um bom chá para a calvície? É o chá-péu.",
    "Por que o jacaré tirou o filho da escola? Porque ele réptil de ano.",
    "Para que servem os óculos vermelhos? Para vermelhor.",
    "O que um tijolo falou para o outro? Há um cimento entre nós.",
    "Como fazer um nó em duas motos? Pega as duas e Yamaha.",
    "Qual campeonato é melhor que aspirina? Liberta-dores.",
    "um amigo diz a outro: Há duas palavras que abrem muitas portas na vida. e o amigo responde. Quais são? e o amigo responde. Puxe e empurre.",
    "O que o Aquaman faz para salvar o mundo? Nada.",
    "a mãe pega uma bexiga e diz para o filho estourar. Qual é o nome do filme? Tó, estoure.",
    "Qual a diferença de 14h para 2h? 12 horas.",
    "Era uma vez um pintinho chamado Relam. Toda vez que chovia, Relam piava.",
    "O que é um pontinho marrom no Brasil em 1500? Pedro Álvares Cabrown.",
    "O que o sal disse para a batata? É nóis na frita!",
    "O médico chega para a paciente e diz: “você só tem uma perna esquerda”. Ela se desespera e diz que precisa das duas pernas. Então, ele finaliza: “E uma perna direita”",
]


prepiada = [
        f"{username}, Piada saindo dor fornooo. ",
        f"tudo bem, {username} , mais foi você que pediu",
        f"{username}, Eu tenho uma piada que vai fazer você rir. ",
        f"{username}, Você está preparado para rir? Porque eu tenho uma piada ótima!",
        f"{username}, Eu tenho uma piada que eu tenho certeza que você vai gostar. ",
        f"perai {username}, essa piada é tão engraçada que eu não consigo parar de rir só de pensar nela. hahahaha. pronto. ",
        f"{username}, Eu tenho uma piada é tão ruim que é boa. você vai adorar. ",
        f"{username} você está pronto para rir? Porque eu estou prestes a contar a melhor piada que você já ouviu. ",] 


listapresentar = [
    f"olá {username}, me chamo alfa, minha função é ser uma assistente virtual segura, em que tudo oque você me perguntar será processado e respondido do seu computador, sem passar por nenhum outro servidor externo. algumas das coisas que eu posso fazer são: contar piadas, dizer uma curiosidade, abrir aplicativos instalados em seu computador, entre muitas outras.",
    f"olá {username}, prazer em conhece lo, Meu nome é alfa, sou uma assistênte virtual totalmente programada em python, e um dos meus pontos que me diferenciam das outras é que tudo oque eu faço está no seu dispositivo, ou seja, tudo oque é falado é processado e respondido no seu aparelho. uma das coisas que eu posso faze, por exemplo é contar piadas ou também pesquisar algo na wikipédia e até mesmo no youtube, e isso foi só uma amostra do que eu posso fazer  ",
    f"oi, {username}, eu sou a alfa. uma assistente virtual que responde perguntas e faz pesquisas. tudo oque você presisa fazer é: para uma pesquisa basta dizer buscar, procurar, encontre, procure e etcétera. e além disso eu posso tocar uma músîca da sua escolha dizendo. toque ou tocar, mais a sua músîca, e ela será reproduzida no seu navegador. e além dessas funções existe muitas outras como. pedir a cotação de uma moeda, ou, jogar cara ou coroa." 
    ]   


listatocar = [
    f"{username}, tocando, ",
    f"{username}, reproduzindo a musica, ",
    f"{username}, agora a sua favorita, tocando. ",
]


curiosidade = [
    "Existem mais formas de vida vivendo na sua pele do que humanos habitando a Terra. Esta, certamente, é uma das curiosidades do mundo mais impactantes.",
    "Em média, cada pessoa perde 4kg de pele morta em um ano",
    "Charles Osborne teve uma crise de soluços que durou, nada mais nada menos que, 69 anos. Começou em 1922, quando pesava um cerdo para sacrificá-lo e só parou quando ele já tinha 97 anos.",
    "Todas as pessoas que têm olhos azuis têm um mesmo ancestral em comum.",
    "O cérebro é um órgão extraordinário. Isso porque comanda todo o organismo humano. Além disso, é o único que não pode sentir dor.",
    "Geralmente, 30% do sangue bombeado pelo coração vai direto para o cérebro.",
    "A decomposição do corpo humano começa apenas 4 minutos depois da morte.",
    "Respirar pela boca o tempo todo pode causar cáries e modificar o formato da mandíbula. Uma das curiosidades do mundo mais chocantes",
    "Quando você fala para si mesmo, por exemplo, enquanto lê, essa ‘voz’ interior é acompanhada de movimentos muito sutis da laringe.",
    "Beijar um bebê na orelha pode deixá-lo surdo.",
    "As três famílias mais ricas do mundo têm mais dinheiro que a riqueza dos 48 países mais pobres do mundo.",
    "A cerveja não era considerada uma bebida alcoólica na Rússia até o ano de 2011. Afinal, até então era classificada como um refresco.",
    "Os homens são 6 vezes mais propensos a serem atingidos por um raio do que as mulheres.",
    "É mais provável que uma pessoa morra com um coco caindo sobre a sua cabeça do que por um ataque de tubarão.",
    "Estados Unidos, Birmânia e Libéria são os únicos países no mundo que não usam o sistema métrico como padrão de medição.",
    "Cerca de 2.500 pessoas canhotas morrem a cada ano. Porque inúmeros acidentes são causados por equipamentos e ferramentas criadas para destros.",
    "Em 2006, um australiano tentou vender a Nova Zelândia (o país) no eBay. A propósito, o preço chegou a 3 mil dólares. Contudo, quando o sistema do eBay percebeu do que se tratava, seus administradores suspenderam a oferta.",
    "Em Nova York (EUA) é proibido vender uma casa mal assombrada sem avisar ao comprador.",
    "O primeiro telefone móvel inventado custava 3.995 dólares.",
    "No Japão você pode comprar um sorvete com sabor de enguia.",
    "Mais de 1.000.000 de euros são jogados na ‘Fontana de Trevi’ (Itália). De tempos em tempos a prefeitura de Roma recolhe as moedas e doa para a caridade.",
    "Usain Bolt exige que todas as fotografias tiradas dele sejam realizadas na Jamaica, seu país de origem. Assim ele contribui economicamente para o seu país.",
    "A maioria dos sanitários em Hong Kong utilizam água do mar. Isso acontece para conservarem ao máximo a pouca quantidade de água doce que tem disponível.",
    "Leonardo Di Caprio recebeu esse nome porque enquanto sua mãe observava um quadro de Leonardo Da Vinci, na Itália, ele deu um chute em sua barriga. E ela o acontecido como um sinal.",
    "Se você tem tatuagens e vai visitar o Japão, cuidado ao entrar em águas termais, em alguns locais é proibida a entrada de pessoas com tatuagem.",
    "Se algum dia você estiver procurando algo, olhe da direita para a esquerda. Por estarmos acostumados a ler da esquerda para a direita é mais fácil as coisas passarem despercebidas.",
    "Cerca de 2/3 dos habitantes da Terra nunca viram neve na vida.",
    "O único planeta do Sistema Solar que não tem nome de um deus, é o nosso.",
    "No topo do famoso Monte Everest, existe uma cobertura para automóveis.",
    "A rotação da Terra está diminuindo gradualmente. A propósito, ela roda 17 milissegundos mais devagar a cada 100 anos. O que isso quer dizer? Que nossos dias estão cada vez mais longos. Mas nem tanto assim, só conseguiríamos notar daqui a 140 milhões de anos, quando um dia passaria a ter 25 horas.",
    "Cientistas calcularam que se fosse escavado um túnel através do centro da Terra, e uma pessoa pulasse ali dentro, demoraria 42 minutos e 12 segundos para atravessá-lo por completo.",
    "A Terra é o planeta mais denso do Sistema Solar.",
    "Em 1033, foi registrada uma impressionante temperatura. Afinal, não é todo dia que os termômetros batem os 136ºC.",
    "A Ilha de Socotra é tão isolada que, por lá vivem espécies que não são encontradas em nenhum outro lugar do planeta. Não à toa, portanto, ficou conhecida como o lugar mais estranho da Terra.",
    "Em 1923, um tornado de fogo levou 38 mil pessoas à morte, em Tókio.",
    "Muito antes das árvores, a Terra era coberta por cogumelos gigantes.",
    "De toda a vida animal que se desenvolveu no planeta, aproximadamente 80% têm 6 ou mais pernas.",
    "O thaumoctopus mimicus é um polvo capaz de mudar sua cor e imitar a de outros seres marinhos. Em suma, até hoje, são conhecidas 15 cores que ele pode imitar.",
    "Um beija-flor, afinal, pode pesar menos que uma moeda de um centavo.",
    "Se um leão macho se torna o líder do grupo, ele mata todos os filhotes do líder anterior. Pesado né?",
    "As vacas são capazes de definir suas melhores amigas e sofrem com as perdas.",
    "A pele de uma rã dourada venenosa possui toxinas suficientes para matar 100 pessoas.",
    "Ao contrário da crença popular, a cor vermelha não atiça os touros.",
    "Os únicos mamíferos capazes de voar são os morcegos.", "Uma formiga pode carregar até 60 vezes o seu peso.",
    "Uma cobra píton grande é capaz de engolir uma cabra inteira.",
    " Basicamente, um crocodilo do Nilo pode prender a respiração por até 2 horas.",
    "Os pássaros-Lira são famosos por sua capacidade de imitar qualquer som que ouçam. Como, por exeplo, o choro de um bebê, os gritos de um macaco, o alarme de um despertador e até os barulhos que fazem uma máquina de construção.",
    "Os bicho-preguiça, além de lentos, são muito ignorantes. Ademais, às vezes, confundem seus próprios braços com galhos e caem da árvore que estão empoleirados.",
    "Os bebês elefantes usam suas próprias trombas como ‘chupeta’ para se acalmarem.",
    "A cor rosada dos flamingos, afinal, é devido à sua alimentação. Isso porque, na verdade, eles são brancos.",
    "Os gatos não conseguem sentir sabores doce. Então não, seu gatinho não sente o sabor do sorvete que você deu a ele."]


historia = [
    "éra uma noite. quase 10 horas da manhâ. quando o surdo escutou o mudo dizer que o cego viu o alejado correr atrás de um carro parado. e alí perto á 200 kilometros em uma noite fria de 45 graus, em que o sol iluminava a pálida noite. uma velha de 15 anos deitada em um banco de pedra feito de madeira. dizia calada. que preferia morrer do que perder a vida. essa história tem um autor desconhecido. ",
    "tinha três irmãos. o peninha. o gotinha. e o tijolinho. um dia o peninha perguntou á sua mãe. mãe. porque meu nome é peninha? e a mãe responde. pqoeu quando você nasceu caiu uma peninha na sua cabeça. depois disso o gotinha perguntou. mãe. porque eu me chamo gotinha? e a mãe responde. poque quando você nasceu caiu uma gotinha na sua cabeça. e então o tijolinho pergunta. liufshoiuhoi sufgro iua ",
    "havia três irmão. o pum. o calaboca. e o respeito. um dia o pum roubou uma coxinha da padaria. então ele foi preso. e os cala bocaeo respeito foram soltar ele da delegacia. o respeito tinha medo da delegacia. e ficou na esquina. na padaria que tinha cido roubada. e o cala boca foi soltar o pum. chegando lá o policial pergunta. qual o seu nome? cala boca. cadê o respeito muleke? fico ná esquina. e oque você veio fazer aqui? Soltar o pum "

]

prehist = [
    f"{username}. vou lhe contar uma história. . "

]

pospiada = [
    "Sounds\Som Piada 1.mp3",
    "Sounds\Som piada 2.mp3",
]


def main_window():



    sg.theme('darkblue1')
    layout = [
        [sg.Text('olá, Bem vindo a Alpha', right_click_menu= (True, 'GLP'))],
        [sg.Image('images/Alpha Icon.png')],
        [sg.Text('Use palavras-chave para fazer a pergunta , EX: busque, encontre, ')],
        [sg.Text('ache entre outros',)],
        [sg.Button(key= 'Ligar Alpha' , button_text='Ligar Alpha' , border_width= 0,),sg.Button('Fechar Alpha', border_width= 0), sg.Button('Configurações', border_width= 0)],
        ]
    
    return sg.Window('AVA  Alpha - Tela Inicial', layout= layout, finalize= True,font=('Nasalization rg', 11))


def config_window():
    sg.theme('darkblue1')

    layout2 = [
        [sg.Text('escreva abaixo por oque que você quer que a alpha te chame',)],
        [sg.Input(key= 'nameuser'), sg.Button('salvar', border_width= 0,)],
        [sg.Button('Salvar E Voltar', border_width= 0,)]
    ]

    return sg.Window('AVA  Alpha - Configurações', layout= layout2, finalize= True,font=('Nasalization rg', 11))

janela_inicial, janela_config = main_window(), None


while True:

    window,event,value = sg.read_all_windows()

    if window == janela_inicial and event == sg.WIN_CLOSED:
        sys.exit()


    elif window == janela_config and event == sg.WIN_CLOSED:
        sys.exit()


    if window == janela_inicial and event == 'Configurações':
        janela_config = config_window()

    

    elif window == janela_config and event == 'Salvar E Voltar':

        janela_config.hide()
        janela_inicial.un_hide()


    if window == janela_inicial and event == 'Ligar Alpha':
        iafala('pode falar')
        comando_voz_usuario()

        with open('Responce.dll', "r") as ablabla:
            ablablabla = ablabla.read()
            print(ablablabla)

        with open('ask.dll',  "r") as bleu:
            pergunta = bleu.read()
        sg.popup(f'{username}: {pergunta} \n \n Alpha: \n {ablablabla}', title= 'Resposta', auto_close_duration= 20, auto_close= True,font=('Nasalization rg', 11))
        


    if window == janela_inicial and event == 'Fechar Alpha':
        pyaudio('Sounds\Error.wav')
        iafala('alguem me desligou! até')
        sys.exit()


    elif window == janela_config and event == 'salvar':
        try:
            with open('username.dll', "w") as usname:
                susname = (value['nameuser'])
                usname.write(susname)
            pyaudio('Sounds\save.wav')
            iafala(f'Nome de usuário salvo como: {susname}') 
        except:
            pyaudio('Sounds\Error.wav')
            iafala(f'não foi possível salvar seu nome de usuário como. {susname}, verifique os arquivos ou tente novamente mais tarde.') 

    

    if window == janela_inicial and event == 'GLP':
        pyaudio('Sounds\..mp3')

        



                    