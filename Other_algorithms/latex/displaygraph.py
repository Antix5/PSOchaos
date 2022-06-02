subtitles = ["Evolution of ERJ depending in regards to the number of dimensions | Dimension 2",	
            "Evolution of ERJ depending in regards to the number of dimensions | Dimension 5",
            "Evolution of ERJ depending in regards to the number of dimensions | Dimension 10",
            "Evolution of ERJ depending in regards to the number of dimensions | Dimension 50",
            "Evolution of ERJ depending in regards to the number of dimensions | Dimension 100",
            "Evolution of ERJ depending on the dimension",
            "Evolution of ERJ depending on the number of K | Rastringin",
            "Evolution of ERJ depending on the number of K | Booth",
            "Evolution of the avarage k depending on the value of P | Rastringin",
            "Evolution of the avarage k depending on the value of P | Booth"]

commentaries = ["The ERJ is: 0.017894385917374578",
"The ERJ is: 2.9848775024190743",
"The ERJ is: 4.014968644746489",
"The ERJ is: 205.75578243398908",
"The ERJ is: 562.2491221239749",
"The higher the dimension, the higher ERJ is",
"Oservation of ERJ and K for the rastringin function",
"Oservation of ERJ and K for the booth function",
"Oservation of the avarage k and P for the rastringin function",
"Oservation of the avarage k and P for the booth function"]


for k in range(10):
    print("\\begin{frame}")
    print("\\frametitle{\\color{velvet} Study of the impact of the parameters on the convergence}")
    print("\\framesubtitle{" + subtitles[k] + "}")
    print(commentaries[k])
    print("\\includegraphics[scale=0.5]{Graphs/"+str(k+1)+".png}")
    print("\\end{frame}")
