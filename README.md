# CNN Shape versus Texture

**Nanthawat Sirisuksumran — Individual Deep Learning experiment**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaxamoto-blip/CNN/blob/main/Shape_Texture_Experiment.ipynb)

[Notebook พร้อมรัน](Shape_Texture_Experiment.ipynb) · [ผลดิบทุก seed](results/per_run_metrics.csv) · [คำทำนายก่อนรัน](PREREGISTRATION.md) 

## คำถามและการทดลอง

เมื่อรูปร่างบอกคลาสหนึ่ง แต่พื้นผิวบอกอีกคลาส CNN จะเลือกตามอะไร?
ใช้ภาพเลขลายมือสาธารณะ 1,797 ภาพสร้างลายสังเคราะห์ 10 แบบ แล้วฝึก CNN โครงสร้างเดียวกันจากศูนย์สามเงื่อนไข:

- **Plain:** เลขที่ไม่มีลาย
- **Correlated:** texture ID ตรงกับ digit label
- **Randomized:** texture ID สุ่มอิสระจาก digit label

ใช้ seed 11, 22, 33 รวมเก้าโมเดล แต่ละโมเดลฝึก 50 epochs เท่ากัน
คู่ Correlated และ Randomized ใช้ภาพต้นฉบับ ค่าเริ่มต้น ลำดับ batch flags ภาพปกติ และ phase ที่จับคู่กัน
ทุกเงื่อนไขมีภาพ grayscale ปกติประมาณ 25% ต่อ epoch โดยสุ่ม Bernoulli

แบ่ง train/validation/test ที่ระดับภาพต้นฉบับเป็น **1,077/360/360** ก่อนสร้างภาพดัดแปลง
บนรูปร่างเดียวกัน ทุกลายมี mask และ histogram ของ intensity เหมือนกันทุกค่า จึงลดคำอธิบายจากความสว่างและจำนวนพิกเซล
ภาพ conflict ใช้ครบเก้าลายที่ขัดกับคลาสของภาพต้นฉบับแต่ละภาพ รวม **3,240 ภาพต่อโมเดล**

![ตัวอย่างภาพปกติ ภาพ aligned และภาพ conflict](https://github.com/user-attachments/assets/e4da333c-b704-4223-9918-c81edb089b53)

## ผลจากการรันจริง

ค่าในตารางเป็น **ร้อยละ ค่าเฉลี่ย ± sample SD จากสาม seed**

| เงื่อนไข | Normal accuracy | Conflict ตามรูปร่าง | Conflict ตามลาย |
|---|---:|---:|---:|
| Plain | 97.59 ± 0.32 | 90.33 ± 1.81 | 1.17 ± 0.08 |
| Correlated | 94.91 ± 1.89 | 88.83 ± 1.44 | 2.65 ± 0.54 |
| Randomized | 97.13 ± 1.12 | 95.09 ± 0.59 | 0.61 ± 0.02 |

![ความแม่นยำบนภาพปกติและสัดส่วนการตัดสินภาพ conflict](https://github.com/user-attachments/assets/b2fade3c-0e1a-42c0-b3c0-0f7faf79f73f)

**ผลหลัก:** ใน CNN และข้อมูลที่ทดลอง ทุกเงื่อนไขตอบตามรูปร่างเป็นหลัก
Randomized เพิ่ม conflict shape agreement จาก Correlated เฉลี่ย **6.27 จุดเปอร์เซ็นต์**
ช่วงความเชื่อมั่น 95% แบบ cluster bootstrap อยู่ที่ **5.00–7.55 จุดเปอร์เซ็นต์**
โดยสุ่มที่ระดับภาพต้นฉบับและเก็บภาพดัดแปลงเก้าภาพไว้ด้วยกัน
CI นี้มีเงื่อนไขบน split และโมเดลที่ฝึกแล้ว ไม่รวมความไม่แน่นอนจาก split ใหม่หรือผู้เขียนตัวเลข

Conditional shape bias = S/(S+T) ไม่รวมคำตอบ Other
จึงรายงาน S/N, T/N, Other/N และ coverage ควบคู่ใน [ตารางผลครบ](results/summary.json)
การดีขึ้นอาจรวมความทนต่อภาพดัดแปลง ไม่ใช่การเปลี่ยนจาก texture ไป shape ทุกคำตอบ
ผลนี้ไม่ยืนยันว่าโมเดลใช้ global outline แยกจาก local strokes และไม่สรุปแทน CNN ทุกชนิด

## รันด้วย Colab

เปิด [Shape_Texture_Experiment.ipynb](Shape_Texture_Experiment.ipynb) ด้วยปุ่ม **Open In Colab** ด้านบน แล้วเลือก **Run all**
Notebook มีโค้ดครบในตัว สร้างไฟล์โครงการและผลอ้างอิงให้เอง ใช้ CPU และไม่ต้องใช้ API key
ผลอ้างอิงมาจากสคริปต์ที่รันจริงบน CPU ไม่ใช่การอ้างว่าเซลล์ใน Colab ถูก execute ไว้แล้ว

## รันบนเครื่องหรือใน VS Code

แนะนำ Python 3.12 แล้วรันจากโฟลเดอร์ repo:

```bash
python -m pip install -r requirements.txt
python experiment.py
python make_figures.py
python experiment.py --verify
python verify_retraining.py
python positive_control.py
```

`experiment.py` ฝึกเก้าโมเดลเมื่อยังไม่มี checkpoints
หากมี checkpoints อยู่แล้วจะใช้โมเดลเดิม โดย `verify_retraining.py` ฝึกทั้งเก้าโมเดลใหม่ในโฟลเดอร์ชั่วคราวเพื่อเทียบกับรอบก่อน
ภาพทดสอบและ checkpoints สร้างขึ้นจากโค้ดเมื่อรัน ไม่ต้องดาวน์โหลด pretrained weights
การรันใหม่อาจเขียนผลใน `results/` ทับค่าที่ checkout มา จึงใช้ Git เปรียบเทียบกับผลอ้างอิงได้

การตรวจในสภาพแวดล้อมอ้างอิงผ่านแล้ว: น้ำหนักทุกโมเดล logits predictions และ metrics ตรงกันทุกค่า
ข้ามแพลตฟอร์มหรือเวอร์ชันอาจไม่ตรงระดับ bitwise จึงบันทึก [สภาพแวดล้อม](results/environment.json) และ [หลักฐานการตรวจ](results/full_retraining_check.json) ไว้

## หลักฐานและไฟล์สำคัญ

| ไฟล์ | เนื้อหา |
|---|---|
| [config.json](config.json) | พารามิเตอร์และ seed ที่ใช้จริง |
| [PREREGISTRATION.md](PREREGISTRATION.md) | คำทำนายและแผนที่บันทึกก่อนฝึก |
| [preregistration_record.json](preregistration_record.json) | เวลาบันทึกและ hash ของแผน ไม่ใช่การลงทะเบียนภายนอก |
| [data/split.json](data/split.json) | ดัชนีต้นฉบับและจำนวนคลาสของทุก split |
| [data/audit.json](data/audit.json) | แหล่งข้อมูล license และการตรวจภาพซ้ำข้าม split |
| [results/training_records.json](results/training_records.json) | ประวัติการฝึกและค่าเริ่มต้นของเก้าโมเดล |
| [results/per_run_metrics.csv](results/per_run_metrics.csv) | ผลดิบครบทุก seed และทุก test view |
| [results/paired_effect.json](results/paired_effect.json) | ความต่างที่จับคู่ seed และ cluster bootstrap CI |
| [EXPLORATORY_ADDENDUM.md](EXPLORATORY_ADDENDUM.md) | การตรวจเพิ่มที่ประกาศหลังเห็นผลหลัก |

การตรวจ texture บน disk ที่ไม่มีรูปร่างเลขได้ 10% ในโมเดลหลัก แต่เป็น out of distribution
จึงเพิ่ม positive control แบบ **exploratory หลังผลหลัก** โดยฝึก CNN เดิมบน disk ให้ทาย texture โดยตรง
ได้ 100% ทั้งสาม seed ตาม [ผลตรวจ](results/positive_control.json)
ผลนี้แสดงว่าลายเรียนรู้ได้เมื่อฝึกตรงงาน และไม่เปลี่ยนคำทำนายหรือผลหลัก

## แหล่งข้อมูลและข้อจำกัด

ใช้ **Optical Recognition of Handwritten Digits** ของ E. Alpaydin และ C. Kaynak (1998)
ผ่าน `sklearn.datasets.load_digits` จำนวน 1,797 ภาพ ซึ่งเป็นสำเนา test subset เดิมของ UCI
ไม่ใช่ UCI ทั้งชุด 5,620 ภาพ ภาพดัดแปลงและลายสังเคราะห์สร้างสำหรับการทดลองนี้

- [Dataset และ DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49)
- Dataset license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
- [Loader documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- แนวคิด cue conflict: [Geirhos et al., ICLR 2019](https://arxiv.org/abs/1811.12231)
- ข้อควรระวังเรื่อง feature reliance: [Burgert et al., NeurIPS 2025](https://arxiv.org/abs/2509.20234)

ข้อจำกัดคือเลขความละเอียดต่ำ ลายสังเคราะห์ หนึ่ง split และสาม seed
ไม่ได้แบ่งตามผู้เขียน และไม่แยก global shape จาก local strokes

## การใช้ AI

ใช้ ChatGPT ช่วยตั้งคำทำนาย ออกแบบการทดลอง เขียนและรันโค้ด วิเคราะห์ ตรวจการทำซ้ำ และร่างรายงาน
ประเด็นที่ตรวจแก้คือการอ่าน conditional shape bias โดยไม่ดู Other และการตีความ disk ที่เป็น out of distribution
รายละเอียดการใช้ AI และข้อจำกัดรวมอยู่ในรายงาน PDF ที่ส่งประกอบ ผู้ส่งเป็นผู้รับผิดชอบผลงาน
