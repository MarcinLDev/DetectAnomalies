# AI – Predykcja awarii sieci wodno-kanalizacyjnej

Projekt prezentuje interaktywny dashboard Streamlit do oceny ryzyka awarii w sieciach wodno‑kanalizacyjnych z wykorzystaniem modeli uczenia maszynowego (Random Forest). Aplikacja umożliwia analizę danych, predykcję ryzyka, symulacje „co‑jeśli” oraz eksport raportu PDF.

## Kluczowe funkcje

- Klasyfikacja ryzyka awarii: niskie / średnie / wysokie
- Sandbox „co‑jeśli” (interaktywna symulacja)
- Wizualizacje wpływu cech na wynik modelu
- Eksport do PDF i CSV
- Live demo (opcjonalnie)

### Ustawienia + Podsumowanie ryzyka
![Podsumowanie ryzyka](assets/screens/dashboard_1.png)

### Czynniki wpływające na ryzyko
![Dashboard – wykres ważności cech](assets/screens/dashboard_2.png)

### Tryb „co‑jeśli” + Miernik ryzyka
![Dashboard – tryb symulacji](assets/screens/dashboard_3.png)

### Tabela TOP elementów
![Dashboard – TOP tabela](assets/screens/dashboard_4.png)

## Technologie

- Python 3.10+
- Streamlit
- Scikit-learn (Random Forest)
- Plotly
- Pandas
- ReportLab (PDF)
- Kaleido (eksport wykresów do PNG)

## Uruchomienie

1. Zainstaluj zależności:

```bash
pip install -r requirements.txt
