# AI – Predykcja awarii sieci wodno-kanalizacyjnej

Projekt prezentuje **aplikację demonstracyjną AI** służącą do oceny ryzyka awarii
w sieciach wodno-kanalizacyjnych z wykorzystaniem modeli uczenia maszynowego.

Aplikacja została zbudowana w formie **interaktywnego dashboardu Streamlit**
i umożliwia analizę danych, predykcję ryzyka, symulacje typu *„co-jeśli”*
oraz generowanie raportów.

Projekt ma charakter **case study / proof-of-concept** i skupia się na
praktycznym wykorzystaniu modelu ML w aplikacji dla użytkownika biznesowego.

---

## Problem

Sieci wodno-kanalizacyjne są narażone na awarie wynikające z wielu czynników,
takich jak wiek infrastruktury, warunki środowiskowe czy intensywność eksploatacji.

Celem projektu jest:
- wczesna identyfikacja elementów infrastruktury o podwyższonym ryzyku awarii
- wsparcie decyzji operacyjnych i planowania działań prewencyjnych
- umożliwienie użytkownikowi eksploracji scenariuszy „co-jeśli”

---

## Opis rozwiązania

Projekt wykorzystuje **model Random Forest** do klasyfikacji ryzyka awarii
na trzy poziomy:
- niskie
- średnie
- wysokie

Model został osadzony w aplikacji webowej, która:
- umożliwia interaktywną analizę danych wejściowych
- prezentuje wpływ cech na wynik predykcji
- pozwala symulować zmiany parametrów i obserwować wpływ na ryzyko
- generuje raporty w formacie PDF i CSV

---

## Kluczowe funkcje aplikacji

- Klasyfikacja ryzyka awarii (low / medium / high)
- Interaktywny tryb „co-jeśli” (sandbox symulacyjny)
- Wizualizacja wpływu cech na wynik modelu
- Tabela TOP elementów o najwyższym ryzyku
- Eksport wyników do PDF i CSV
- Czytelny dashboard dla użytkownika nietechnicznego

---

## Przykładowe widoki aplikacji

### Podsumowanie ryzyka
![Podsumowanie ryzyka](assets/screens/dashboard_1.png)

### Czynniki wpływające na ryzyko
![Ważność cech](assets/screens/dashboard_2.png)

### Tryb „co-jeśli” i miernik ryzyka
![Tryb symulacji](assets/screens/dashboard_3.png)

### Tabela TOP elementów
![Tabela TOP](assets/screens/dashboard_4.png)

---

## Technologie

- Python 3.10+
- Streamlit
- Scikit-learn (Random Forest)
- Pandas
- Plotly
- ReportLab (generowanie PDF)
- Kaleido (eksport wykresów do PNG)

---

## Uruchomienie projektu

1. Zainstaluj zależności:

```bash
pip install -r requirements.txt

streamlit run app.py
```

## Zakres projektu

Projekt celowo koncentruje się na warstwie aplikacyjnej i modelowej.

W zakresie projektu:

- trening i inferencja modelu ML
- logika biznesowa predykcji ryzyka
- interfejs użytkownika i wizualizacja wyników
- raportowanie
- Poza zakresem projektu:
- produkcyjne pipeline’y danych (ETL / Airflow)
- wdrożenie chmurowe
- monitoring i MLOps

## Uwagi końcowe

* Projekt ma charakter demonstracyjny i edukacyjny
* Dane wykorzystywane w projekcie nie zawierają informacji wrażliwych
* Repozytorium stanowi przykład praktycznego użycia modelu ML w aplikacji
