# أمثلة عملية لاستخدام مشروع NeuroPi

## 🚀 البدء السريع

### 1. تشغيل المشروع
```bash
# تفعيل البيئة الافتراضية
python -m venv neuropi_env
source neuropi_env/bin/activate  # Linux/Mac
# أو
neuropi_env\Scripts\activate     # Windows

# تثبيت المتطلبات
pip install -r requirements.txt

# تشغيل قاعدة البيانات
python manage.py migrate

# إنشاء مستخدم إداري
python manage.py createsuperuser

# تشغيل الخادم
python manage.py runserver
```

### 2. الوصول للنظام
- الصفحة الرئيسية: `http://localhost:8000/`
- لوحة الإدارة: `http://localhost:8000/admin/`
- تطبيق التجارب: `http://localhost:8000/trials/`
- تطبيق BCI: `http://localhost:8000/BCI/`

## 📊 أمثلة على استخدام التطبيقات

### تطبيق trials - جمع بيانات الكلمات

#### إنشاء مجموعة كلمات جديدة
```python
# في Django Shell: python manage.py shell
from trials.models import WordSet, Word

# إنشاء مجموعة كلمات
word_set = WordSet.objects.create(
    name="كلمات عربية أساسية",
    description="مجموعة من الكلمات العربية الشائعة"
)

# إضافة كلمات
words = ["بيت", "سيارة", "كتاب", "قلم", "ماء"]
for word_text in words:
    Word.objects.create(
        word_set=word_set,
        text=word_text
    )
```

#### تشغيل تجربة كلمات بصرية
```python
# في views.py
def start_visual_trial(request):
    if request.method == 'POST':
        participant_name = request.POST.get('participant_name')
        word_set_id = request.POST.get('word_set')
        
        # إنشاء جلسة جديدة
        session = VisualTrialSession.objects.create(
            participant_name=participant_name,
            word_set_id=word_set_id,
            word_display_duration=3000,  # 3 ثواني
            rest_duration=2000,          # ثانيتان راحة
            repetitions_per_word=10      # 10 تكرارات
        )
        
        return redirect('run_visual_trial', session_id=session.id)
```

### تطبيق motor_imagery - التخيل الحركي

#### إعداد جلسة تخيل حركي
```python
from motor_imagery.models import MotorImagerySession

# إنشاء جلسة جديدة
session = MotorImagerySession.objects.create(
    participant_name="أحمد محمد",
    session_name="جلسة تدريب أولى",
    imagery_duration=4000,    # 4 ثواني تخيل
    cue_duration=2000,        # ثانيتان إشارة
    rest_duration=2000,       # ثانيتان راحة
    trials_per_class=20       # 20 تجربة لكل فئة
)

print(f"إجمالي التجارب: {session.total_trials}")
print(f"المدة المتوقعة: {session.estimated_duration_minutes:.1f} دقيقة")
```

#### جمع بيانات EEG أثناء التخيل
```python
# في data_collection.py
class MotorImageryDataCollector:
    def __init__(self, session):
        self.session = session
        self.eeg_device = EmotivEEG()
        
    def run_trial(self, imagery_class):
        # عرض الإشارة
        self.show_cue(imagery_class)
        time.sleep(self.session.cue_duration / 1000)
        
        # بدء تسجيل EEG
        self.eeg_device.start_recording()
        
        # فترة التخيل
        self.show_imagery_instruction(imagery_class)
        time.sleep(self.session.imagery_duration / 1000)
        
        # إيقاف التسجيل
        eeg_data = self.eeg_device.stop_recording()
        
        # حفظ البيانات
        self.save_trial_data(imagery_class, eeg_data)
```

### تطبيق BCI - تدريب النماذج

#### تحضير البيانات للتدريب
```python
from BCI.ml_core import EEGDataProcessor
import pandas as pd

# تحميل بيانات من ملف CSV
data = pd.read_csv('Motor_Imagery_data/session_val_10/motor_imagery_20250628_212214.csv')

# معالجة البيانات
processor = EEGDataProcessor()
X, y = processor.prepare_data(
    data=data,
    window_size=2.0,      # نافذة زمنية 2 ثانية
    overlap=0.5,          # تداخل 50%
    channels=['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
              'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
)

print(f"شكل البيانات: {X.shape}")
print(f"عدد الفئات: {len(np.unique(y))}")
```

#### تدريب نموذج ATCNet
```python
from BCI.ml_core import ATCNet, ModelTrainer
import torch

# إنشاء النموذج
model = ATCNet(
    n_channels=14,        # 14 قناة EEG
    n_classes=4,          # 4 فئات (يسار، يمين، قدمين، راحة)
    dropout_rate=0.5
)

# تدريب النموذج
trainer = ModelTrainer(model)
history = trainer.train(
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    batch_size=32,
    learning_rate=0.001
)

# حفظ النموذج
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'n_channels': 14,
        'n_classes': 4,
        'dropout_rate': 0.5
    }
}, 'trained_model.pt')
```

### تطبيق preprocessor - معالجة الإشارات

#### تطبيق المرشحات
```python
from preprocessor.process_eeg import EEGPreprocessor
import numpy as np

# إنشاء معالج الإشارات
preprocessor = EEGPreprocessor(sampling_rate=128)

# تحميل البيانات الخام
raw_data = np.loadtxt('raw_eeg_data.csv', delimiter=',')

# تطبيق مرشح تمرير النطاق
filtered_data = preprocessor.bandpass_filter(
    data=raw_data,
    low_freq=8.0,    # تردد منخفض
    high_freq=30.0   # تردد عالي
)

# إزالة تداخل التيار الكهربائي
clean_data = preprocessor.notch_filter(
    data=filtered_data,
    notch_freq=50.0  # 50 Hz في أوروبا، 60 Hz في أمريكا
)

# تطبيق Common Average Reference
car_data = preprocessor.apply_car(clean_data)

print(f"البيانات الأصلية: {raw_data.shape}")
print(f"البيانات المعالجة: {car_data.shape}")
```

## 🔄 سيناريوهات الاستخدام الكاملة

### سيناريو 1: تجربة كلمات بصرية كاملة

```python
# 1. إعداد التجربة
def setup_word_experiment():
    # إنشاء مجموعة كلمات
    word_set = WordSet.objects.create(
        name="تجربة الكلمات العربية",
        description="كلمات للتعرف على النشاط العصبي"
    )
    
    words = ["منزل", "شجرة", "نهر", "جبل", "سماء"]
    for word in words:
        Word.objects.create(word_set=word_set, text=word)
    
    return word_set

# 2. تشغيل التجربة
def run_word_experiment(participant_name, word_set):
    session = VisualTrialSession.objects.create(
        participant_name=participant_name,
        word_set=word_set,
        word_display_duration=3000,
        rest_duration=2000,
        repetitions_per_word=5
    )
    
    # تشغيل جمع البيانات
    collector = VisualTrialDataCollector(session)
    collector.run_complete_session()
    
    return session

# 3. معالجة البيانات
def process_experiment_data(session):
    # تحميل البيانات
    data_file = f"Trials_data/visual_trial_{session.participant_name}/session_{session.id}/visual_trial_data.csv"
    data = pd.read_csv(data_file)
    
    # معالجة الإشارات
    preprocessor = EEGPreprocessor()
    processed_data = preprocessor.full_pipeline(data)
    
    return processed_data
```

### سيناريو 2: تدريب نموذج BCI كامل

```python
# 1. جمع بيانات التدريب
def collect_training_data():
    sessions = []
    participants = ["أحمد", "فاطمة", "محمد", "عائشة"]
    
    for participant in participants:
        session = MotorImagerySession.objects.create(
            participant_name=participant,
            session_name=f"جلسة تدريب {participant}",
            trials_per_class=30
        )
        
        # تشغيل جمع البيانات
        collector = MotorImageryDataCollector(session)
        collector.run_complete_session()
        sessions.append(session)
    
    return sessions

# 2. تحضير البيانات المجمعة
def prepare_combined_data(sessions):
    all_data = []
    all_labels = []
    
    for session in sessions:
        data_file = session.eeg_data_file
        data = pd.read_csv(data_file)
        
        processor = EEGDataProcessor()
        X, y = processor.prepare_data(data)
        
        all_data.append(X)
        all_labels.append(y)
    
    # دمج البيانات
    X_combined = np.concatenate(all_data, axis=0)
    y_combined = np.concatenate(all_labels, axis=0)
    
    return X_combined, y_combined

# 3. تدريب وحفظ النموذج
def train_and_save_model(X, y):
    # تقسيم البيانات
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # تدريب النموذج
    model = ATCNet(n_channels=14, n_classes=4)
    trainer = ModelTrainer(model)
    
    history = trainer.train(X_train, y_train, X_test, y_test)
    
    # حفظ في قاعدة البيانات
    training_model = TrainingModel.objects.create(
        name=f"نموذج BCI {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        user=request.user,
        config={
            'n_channels': 14,
            'n_classes': 4,
            'accuracy': history['val_accuracy'][-1]
        }
    )
    
    # حفظ ملفات النموذج
    model_path = f"trained_models/{training_model.name}.pt"
    torch.save(model.state_dict(), model_path)
    training_model.model_file = model_path
    training_model.save()
    
    return training_model
```

## 🎯 نصائح للاستخدام الأمثل

### 1. جودة البيانات
- تأكد من جودة اتصال جهاز EEG
- تجنب الحركة أثناء التسجيل
- استخدم بيئة هادئة ومريحة

### 2. تحسين النماذج
- اجمع بيانات كافية (100+ تجربة لكل فئة)
- استخدم تقنيات Data Augmentation
- جرب معاملات مختلفة للنموذج

### 3. الأداء والسرعة
- استخدم GPU للتدريب السريع
- قم بتحسين حجم النوافذ الزمنية
- استخدم تقنيات التوازي في المعالجة

---

هذه الأمثلة توضح كيفية استخدام مختلف مكونات مشروع NeuroPi بطريقة عملية وفعالة.
