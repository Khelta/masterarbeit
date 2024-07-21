# Anomalieerkennung mit unüberwachten Autoencodern durch Beschränkung der Trainigsdatenauswahl

Die Ergebnisse können unter [https://seafile.rlp.net/d/3b75c43e41ab4030947d/](https://seafile.rlp.net/d/3b75c43e41ab4030947d/) eingesehen werden.

## Zusammenfassung
Anomalieerkennung beschreibt das Problem Muster innerhalb eines Datensatzes zu finden, die nicht einem zu erwarteten Verhalten entsprechen.
Im Falle der unüberwachten Anomalieerkennung verfügen die Datensätze, auf denen Modelle zur Anomalieerkennung trainiert werden sollen, über keine Klassenlabel. 
Eine Möglichkeit zur unüberwachten Anomalieerkennung besteht in der Verwendung von Autoencodern.
In dieser Arbeit wird ein Ansatz vorgestellt, der während des Trainings die Instanzen anhand ihres Rekonstruktionsfehler von der Gradientenberechnung ausschließt, um so eine bessere Anomalieerkennungsleistung durch Ausschluss der Anomalien zu erhalten.
Für einen Teil der analysierten Datensätze konnte eine bessere Anomalieerkennungsleistung dieses Ansatzes im Vergleich zu etablierten Methoden gezeigt werden.

## Setup
Die benötigten Bibliotheken können via 
```bash
pip install -r ./requirements.txt
```
installiert werden. Für Systeme, die kein Windows verwenden müssen eventuell einige Requirements aus der Liste herausgenommen werden. 
Wenn gewünscht können die Bibliotheken auch händisch installiert werden. 
Relevant sind torch, numpy, pandas, sklearn sowie matplotlib, seaborn für die Plots.

## Run
Um die Ergebnisse zu reproduzieren empfiehlt es sich die run_script.py Datei laufen zu lassen.
Hierbei werden die Ergebnisse für die in der Masterarbeit gegebene Konfiguration erstellt.
Für die Ergebnisse des DeepSVDD bitte folgendes Repository verwenden: [https://github.com/Khelta/Deep-SVDD-PyTorch](https://github.com/Khelta/Deep-SVDD-PyTorch)
Sobald alle Ergebnisse erstellt wurden, kann evaluation.py aufgerufen werden, um die einzelnen Ergebnisse zu sammeln.
Details zum Format finden sich [\*hier\*](https://seafile.rlp.net/d/3b75c43e41ab4030947d/)
Für einzelne Modelle mit einer benutzerdefinierten Konfiguration kann die run.py Datei verwendet werden.
Die verschiedenen Grafiken können in plots.py, roc_curve.py sowie statistical_analysis.py erstellt werden.