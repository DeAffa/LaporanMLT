# Laporan Proyek Machine Learning - Muhammad Daffa Nurahman

## Domain Proyek
Industri otomotif saat ini terus berkembang dengan cepat, sehingga banyak pilihan kendaraan yang berbeda-beda dari segi harga, kapasitas, dan fitur keamanan. Hal ini membuat konsumen dan produsen perlu cara yang cepat dan tepat untuk menilai kualitas atau kelayakan kendaraan. Penilaian secara manual memakan waktu dan bisa bersifat subjektif. Oleh karena itu, diperlukan solusi otomatis menggunakan machine learning yang dapat mengklasifikasikan kendaraan berdasarkan spesifikasi teknisnya.

Dataset Car Evaluation yang dibuat oleh Bohanec dan Rajkovic (1998) banyak digunakan dalam penelitian klasifikasi kendaraan. Dataset ini memuat fitur-fitur seperti harga beli, biaya perawatan, jumlah pintu, kapasitas penumpang, ukuran bagasi, dan tingkat keamanan, yang kemudian diklasifikasikan menjadi kategori kelayakan kendaraan : **unacc**, **acc**, **good**, dan **vgood**. Dengan menggunakan algoritma klasifikasi, model machine learning dapat membantu memprediksi kategori kendaraan secara cepat dan akurat, yang bermanfaat bagi konsumen dan pelaku industri otomotif.

## Business Understanding
Dalam dunia otomotif, konsumen dan produsen harus bisa menilai apakah suatu kendaraan layak digunakan atau tidak, berdasarkan parameter seperti harga, kapasitas, dan keamanan. Namun, proses evaluasi manual membutuhkan waktu dan bisa bersifat subjektif karena tergantung pada opini masing-masing individu.

1.  Problem Statements

Berikut adalah permasalahan utama yang ingin diselesaikan pada proyek ini : 
  -  Bagaimana cara mengklasifikasikan kendaraan ke dalam kelayakan (unacceptable, acceptable, good, very good) secara otomatis dan akurat
  -  Bagaimana membangun model klasifikasi yang mampu bekerja dengan fitur **kategorikal**, serta bisa diinterpretasikan dan ditingkatkan performanya

2.  Goals

Adapun tujuan utama dari proyek ini adalah :
  -  Mengembangkan model machine learning untuk **mengklasifikasikan kelayakan kendaraan** berdasarkan fitur spesifikasi seperti harga, jumlah, pintu, dan tingkat keamanan.
  -  Membandingkan beberapa model klasifikasi untuk **menemukan model terbaik** dengan akurasi tinggi dan kesalahan klasifikasi yang rendah.
  -  Menggunakan metrik evaluasi seperti **akurasi**, **confusion matrix**, dan **F1-Score** untuk mengukur performa model secara objektif.
3.  Solution Statement

Untuk mencapai tujuan klasifikasi kelayakan kendaraan secara otomatis, proyek ini mengusulkan beberapa pendekatan berbasis machine learning :
  -  **Decision Tree Classifier** sebagai baseline : Algoritma ini dipilih karena sifatnya yang mudah dipahami, cepat dan sangat cocok untuk data dengan fitur kategorikal seperti pada dataset kendaraan ini. Decision Tree juga memberikan struktur aturan yang jelas dalam pengambilan keputusan, sehingga sangat bermanfaat untuk interpretasi awal terhadap data.
  -  **Random Forest Classifier** sebagai pembanding : Model ini merupakan pengembangan dari decision tree yang terdiri dari kumpulan pohon keputusan (ensemble), sehingga dapat meningkatkan akurasi sekaligsu mengurangi resiko overfitting. Random Forest dinilai mampu memberikan performa yang lebih stabil dan akuran pada data tabular seperti dataset yang digunakan dalam proyek ini.

Kinerja semua model akan diukur menggunakan beberapa metrik evaluasi, yaitu **akurasi** untuk mengetahui proporsi prediksi yang benar secara keseluruhan, **confusion matrix** untuk melihat distribusi kesalahan antar kelas, serta **F1-Score** untuk menangani kemunkinan ketidakseimbangan antar kategori kelayakan kendaraan. Dengan pendekatan ini, diharapkan dapat diperoleh model klasifikasi kendaraan yang tidak hanya akurat, tetapi juga efisien dan dapat digunakan untuk mendukung pengambilan keputusan secara otomatis.

# Data Understanding
Proyek ini menggunakan dataset bernama <a href="https://archive.ics.uci.edu/dataset/19/car+evaluation"> Car Evaluation Dataset </a> yang berasal dari laman <a href="https://archive.ics.uci.edu/"> UCI Machine Learning Repository </a> yang telah terbagi ke dalam beberapa kategori. Dataset ini bisa diakses dengan tautan sebagai berikut: https://archive.ics.uci.edu/dataset/19/car+evaluation

Dataset ini terdiri dari **1.728 baris data** dan **6 fitur** (atribut) input serta **1 label** (kelas) output. Semua fitur pada dataset ini bersifat **kategorikal**, dan tidak terdapat nilai kosong atau duplikat, sehingga data berada dalam kondisi bersih dan siap digunakan untuk analisis dan pelatihan model.

📊 Deskripsi Variabel (Fitur)

Berikut adalah uraian dari seluruh fitur pada dataset : 

|     Fitur     | Deskripsi                                                                 | Dtype  |
|:-------------:|:--------------------------------------------------------------------------:|:------:|
| `buying`      | Harga beli mobil (kategori: vhigh, high, med, low)                        | object |
| `maint`       | Biaya perawatan mobil (kategori: vhigh, high, med, low)                   | object |
| `doors`       | Jumlah pintu (kategori: 2, 3, 4, more)                                     | object |
| `persons`     | Kapasitas penumpang (kategori: 2, 4, more)                                 | object |
| `lug_boot`    | Ukuran bagasi (kategori: small, med, big)                                 | object |
| `safety`      | Tingkat keselamatan (kategori: low, med, high)                            | object |
| `class`       | Kategori evaluasi kendaraan (target: unacc, acc, good, vgood)             | object |

Beberapa langkah eksplorasi dilakukan untuk memahami data, antara lain : 
-  **Distribusi label target** (`class`) menunjukkan ketidak seimbangan. Kelas `unacc` mendominasi, sedangkan `vgood` dan `good` memiliki jumlah yang jauh lebih sedikit
-  **Visualisasi tiap fitur** seperti `buying`, `maint`, dan `safety` menunjukkan bahwa data terdistribusi cukup merata pada masing-masing nilai kategorikal.
-  **Crosstab antara fitur dan label** seperti `safety` dengan `class` menunjukkan bahwa kendaraan dengan tingkat keselamatan tinggi cenderung mendapatkan evaluasi lebih baik (`vgood` atau `good`)

Visualisasi menggunakan grafik batan juga menunjukkan adanya hubungan yang dapat dimanfaatkan oleh model, terutama pada fitur `safety`, `persons`, dan `buying`, yang tampak memiliki pengaruh terhadap label akhir.

## Data Preparation
Dalam tahap ini, beberapa teknik data preparation diterapkan untuk menyiapkan data sebelum digunakan dalam pelatihan model machine learning. Berikut adalah urutan teknik yang dilakukan : 
1.  **Encoding Data Kategorikal** : Seluruh fitur dalam dataset bersifta kategorikal. Untuk dapat digunakan dalam model machine learning, data kategorikal ini perlu dikonversi ke bentuk numerik. Teknik yang digunakan adalah **Label Encoding**, karena nilai-nilai kategori pada setipa fitur memiliki urutan atau jumlah kategori yang relatif sedikit dan dataset ini sudah cukup bersih. Proses ini dilakukan untuk semua kolom termasuk label (target)
2.  **Membagi Data ke Data Latih dan Data Uji** : Setelah data dikodekan, data kemudian dibagi menjadi **data latih (train)** dan **data uji (test)** menggunakan proporsi 70:30. Pembagian ini bertujuan untuk menguji performa model pada data yang belum pernah dilihat sebelumnya, sehingga hasil evaluasi lebih objektif
3.  **SMOTE Oversampling** : Distribusi kelas target `class` diketahui tidak seimbang, dengan kelas `unacc` yang jauh lebih dominan dibanding `good` dan `vgood`. Untuk mengatasi hal ini, diterapkan teknik **SMOTE (Synthetic Minority Oversampling Technique** dari library `imblearn`. SMOTE hanya diterapkan pada **data latih**, untuk menghindari kebocoran data uji. Hasilnya, jumlah data di setiap kelas menjadi seimbang.

## Modelling
Pada tahap ini, dua model machine learning digunakan untuk menyelesaikan permasalahan klasifikasi jenis kendaraan berdasarkan fitur spesifikasinya : 

1.  **Decision Tree Classifier**
2.  **Random Forest Classifier**

Kedua model dipilih karena memiliki karakteristik yang cocok untuk jenis data pada proyek ini, yaitu data dengan fitur kategorikal yang telah diencoding, dan tidak memerlukan normalisasi atau scaling data. Selain itu :
-  **Decision Tree** adalah model yang bersifat interpretable, cocok untuk memahami pola keputusan karena visualisasi pohon dapat memberikan wawasan logis
-  **Random Forest** adalah model _ensemble_ yang membangun banyak pohon keputusan dan menggabungkan hasil prediksinya dengan metode voting, sehingga memberikan prediksi yang lebih stabil dan akurat.

Proses pemodelan dilakukan sebagai berikut : 
-  Melatih model pada data latih hasil SMOTE (`X-train_bal`, `y_train_bal`)
-  Menguji performa model menggunakan data uji (`Xtest`, `y_test`)
-  Mengevaluasi performa menggunakan metrik **akurasi**, **precision**, **recall**, dan **F1-Score**

Berikut adalah parameter awal yang digunakan :
| Model             | Parameter          | Alasan Pemilihan                                                                                         |
|-------------------|--------------------|-----------------------------------------------------------------------------------------------------------|
| **Decision Tree** | `criterion='gini'` | Digunakan untuk mengukur kualitas split dengan metode pengukuran ketidakmurnian (*impurity*) yang umum. |
|                   | `max_depth=None`   | Tidak membatasi kedalaman pohon untuk menghindari kehilangan informasi awal (nilai default).             |
|                   | `random_state=42`  | Menjaga konsistensi hasil antar percobaan.                                                               |
| **Random Forest** | `n_estimators=100` | Jumlah 100 pohon dipilih sebagai nilai default yang sering memberikan hasil stabil tanpa overfitting.    |
|                   | `max_depth=None`   | Tidak dibatasi untuk awal eksperimen, memungkinkan setiap pohon tumbuh maksimal.                         |
|                   | `random_state=42`  | Untuk reprodusibilitas hasil.                                                                            |

Kelebihan dan Kekurangan dari masing-masing model :
| Model             | Kelebihan                                                                                      | Kekurangan                                                                           |
|-------------------|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| **Decision Tree** | Mudah dipahami dan divisualisasikan. Cepat untuk pelatihan dan prediksi.                      | Mudah overfitting jika tidak dibatasi kedalamannya.                                 |
| **Random Forest** | Lebih akurat, mengurangi overfitting dengan rata-rata banyak pohon. Cocok untuk data kategori. | Lebih kompleks dan membutuhkan lebih banyak sumber daya. Sulit untuk diinterpretasi. |

Setelah dilakukan pelatihan dan evaluasi pada kedua model, didapatkan bahwa :
-  **Random Forest Classifier** memberikan performa yang lebih tinggi pada hampir semua metrik evaluasi dibandingkan Decision Tree
-  Model ini lebih stabil terhadap variasi data dan mengurangi resiko overfitting karena menggunakan ensemble beberapa pohon keputusan
   
## Evaluation
Pada tahap evaluasi ini, digunakan beberapa metrik yang umum digunakan dalam permasalahan klasifikasi multikelas, yaitu **accuracy**, **precision**, **recall**, dan **F1-Score**. Metrik **accuracy** mengukur seberapa banyak prediksi yang benar dari seluruh data uji. Namun, karena klasifikasi ini melibatkan banyak kelas, maka penting juga untuk melihat **precision**, yaitu seberapa tepat model dalam memprediksi suatu kelas, dan **recall** yaitu seberapa baik model dalam mengenali semua data yang benar-benar termasuk dalam kelas tersebut.
Kedua metrik tersebut dilengkapi dengan **F1-Score**, yang merupakan rata-rata harmonis dari precision dan recall, serta sangat bermanfaat ketika terdapat kemungkinan ketidakseimbangan antarkelas.

Hasil Evaluasi menunjukkan bahwa model **Random Forest** memiliki kinerja yang lebih baik dibandingkan **Decision Tree** pada semua metrik yang digunakan. Model Decision Tree memiliki tingkat akurasi sekitar 92%, dengan precision, recall, dan F2-Score rata-rata pada kisaran 90 hingga 92% untuk setiap kelas. Sementara itu, Random Forest mencatatkan akurasi sekitar 96%, dengan precision, recal, dan F1-Score di kisaran 95 hingga 97%. Dengan performa yang lebih stabil dan akurat, Random Forest dipilih sebagai model terbaik dalam proyek ini.

Berikut adalah ringkasan metrik masing-masing model : 
| Metrik    | Decision Tree | Random Forest |
|----------|----------------|----------------|
| Accuracy  | ~92%           | **~96%**        |
| Precision | 91%–93%        | **95%–97%**     |
| Recall    | 90%–92%        | **95%–97%**     |
| F1-Score  | 90%–92%        | **95%–97%**     |

Jika dikaitkan dengan konteks data dan **problem statement**, yaitu mengklasifikasikan jenis evaluasi kendaraan berdasarkan atribut seperti harga, keamanan, dan kapasitas -- maka penggunaan model dengan akurasi dan stabilitas tinggi sangat penting agar hasil klasifikasi benar-benar bisa digunakan dalam proses pengambilan keputusan, misalnya dalam menyusun segmentasi pasar, menentukan kelayakan kendaraan untuk kelompok pengguna tertentu, atau dalam sistem rekomendasi kendaraan otomatis

Dengan performa yang lebih akurat, **Random Forest dipilih sebagai model terbaik** dalam proyek ini. Model ini mampu memberikan hasil prediksi yang konsisten dan dapat diandalkan dalam berbagai kelas, yang sangat penting dalam konteks klasifikasi kendaraan dengan berbagai kategori target. Akurasi dan presisi yang tinggi akan berdampak langsung pada efisiensi sistem dan kepuasan pengguna di sisi aplikasi nyata dari proyek ini.
