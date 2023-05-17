# hert_attack

## Входные параметры для модели:  
  - Возраст пациента
  - Пол пациента
  - Тип боли в груди:
      Значение 1: типичная стенокардия
      Значение 2: атипичная стенокардия
      Значение 3: неангинальная боль
      Значение 4: бессимптомный
  - Артериальное давление в состоянии покоя (в мм рт. ст.)
  - Холескорал в мг / дл, полученный через датчик ИМТ
  - Уровень сахара в крови натощак > 120 мг / дл
      1 = истина
      0 = ложь
  - Результаты электрокардиографии в состоянии покоя
      0 = нормальное
      1 = наличие аномалии зубца ST-T (инверсия зубца Т и/или подъем или депрессия зубца ST > 0,05 мВ)
      2 = показывает вероятную или определенную гипертрофию левого желудочка по критериям Эстеса
  - Достигнута максимальная частота сердечных сокращений
  - Стенокардия, вызванная физической нагрузкой 
      1 = да
      0 = нет
  - Депрессия ST, вызванная физическими упражнениями по сравнению с отдыхом
  - Наклон пикового сегмента упражнения ST
      0 = без наклона
      1 = плоский
      2 = нисходящий наклон
  - Количество крупных судов (0-3)
  - Талассемия
      0 = null
      1 = исправленный дефект
      2 = нормальный
      3 = обратимый дефект

## На выходе: Диагностика заболеваний сердца (статус ангиографического заболевания)
  0: < сужение диаметра 50%. Меньшая вероятность сердечных заболеваний
  1: > сужение диаметра 50%. Больше шансов на сердечные заболевания

## Точность предсказания   0,91
