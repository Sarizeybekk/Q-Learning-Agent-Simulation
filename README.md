# Q-Learning Agent Simulation

Bu proje, **Q-Learning** algoritmasını kullanarak bir ajanı bir ızgarada hareket ettiren bir simülasyon sağlar. Ajan, belirli bir hedefe ulaşmak için ödül ve ceza sistemine göre öğrenir. Proje, ajanın **Q-Tablo** kullanarak hangi eylemleri yapması gerektiğini öğrenmesini sağlar ve her adımda bu eylemleri gerçekleştirir.

### Özellikler
- Q-Learning algoritması kullanılarak ajan öğrenir.
- Ajan, bir ızgarada hareket eder ve hedefe ulaşmak için adımlar atar.
- Ajanın hareketleri `up`, `down`, `left`, `right`, `up-right`, `up-left`, `down-right`, `down-left` gibi yönlerde çapraz hareketleri de içerir.
- Engel ve hedef simülasyonu.
- Eğitim sürecinde ödüller ve ceza değerleriyle ajan yönlendirilir.

### Proje Yapısı

- **`game.py`**: Q-Learning algoritmasını içeren ana simülasyon dosyası.
- **`requirements.txt`**: Proje için gerekli Python kütüphanelerini içerir.
- **`README.md`**: Proje hakkında bilgi ve kullanım talimatları.

### Gerekli Kütüphaneler

Bu projede kullanılan bazı temel Python kütüphanelerini yüklemeniz gerekir. Kütüphaneleri yüklemek için şu komutu kullanabilirsiniz:

```bash
pip install -r requirements.txt
```

```bash
python game.py
```
<img width="419" alt="Ekran Resmi 2025-05-04 22 50 48" src="https://github.com/user-attachments/assets/8b03fd36-a6b4-4b9b-bda3-a9a3bbff129f" />
