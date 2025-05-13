# Laporan Proyek Machine Learning - Muhammad Daffa Nurahman

## Domain Proyek
  Menurut data Kementerian Lingkungan Hidup dan Kehutanan Republik Indonesia, timbulan sampah domestik nasional mencapai sekitar 67,8 juta ton per tahun, tetapi hanya sekitar 11 % yang terkelola melalui daur ulang formal. Banyak Tempat Pembuangan Akhir (TPA) di kabupaten/kota beroperasi di atas 80 % kapasitas desain, menimbulkan risiko pencemaran dan masalah kesehatan masyarakat [1].

  Penelitian Putra et al. menunjukkan bahwa pemilahan manual sampah di tingkat rumah tangga di Jakarta memiliki tingkat kesalahan hingga 25 %, yang mengurangi efisiensi daur ulang dan meningkatkan kontaminasi limbah organik serta anorganik [2]. Hal ini menegaskan perlunya sistem klasifikasi otomatis berbasis machine learning untuk meningkatkan akurasi dan kecepatan sortir.

  Beberapa studi di Indonesia telah mengembangkan solusi ML/CV untuk klasifikasi sampah:
  
- Kadyanan et al. mengimplementasikan Convolutional Neural Network (CNN) untuk klasifikasi sampah padat—termasuk organik, plastik, kertas, dan logam—pada sampel lapangan di Bali, mencapai akurasi hingga 95,2 % tanpa memerlukan lisensi berbayar [3].

- Sari dan Prasetyo memanfaatkan MobileNetV2 dalam aplikasi mobile untuk membedakan sampah organik dan anorganik di Yogyakarta, dengan akurasi 90,2 % dan waktu inference rata rata 120 ms per citra [4].

- Tim Telkom University mengimplementasikan YOLOv5 pada CCTV untuk deteksi sampah plastik di daerah aliran sungai, memperoleh akurasi hingga 84 % dalam kondisi real time [5].

Berdasarkan data dan hasil studi tersebut, penelitian klasifikasi sampah berbasis machine learning di Indonesia penting untuk:
1.	Mengotomasi proses sortir dengan akurasi tinggi dan waktu inferensi rendah.
2.	Meningkatkan tingkat daur ulang nasional melalui pemilahan yang lebih akurat.
3.	Mengurangi beban operasional TPA dan dampak negatif lingkungan.

## Business Understanding
Klasifikasi sampah merupakan salah satu langkah penting dalam mendukung pengelolaan lingkungan yang berkelanjutan. Pemisahan manual membutuhkan waktu dan sumber daya manusia yang tidak sedikit, serta rentan terhadap kesalahan klasifikasi. Oleh karena itu, penerapan teknologi berbasis _Image classification_ melalui deep learning menjadi solusi yang menjanjikan dalam proses otomatisasi pemilahan sampah.

1.  Problem Statements
Berikut adalah permasalahan utama yang ingin diselesaikan pada proyek ini : 
- Pemilahan sampah secara manual tidak efisien dan memakan banyak waktu
- Gambar sampah memiliki latar belakang dan bentuk yang bervariasi, sehingga sulit untuk diklasifikasikan secara akurat oleh sistem otomatis tanpa pendekatan yang akurat
- Keterbatasan sumber daya untuk membangun sistem klasifikasi dari nol mengharuskan penggunaan pendekatan berbasis transfer learning

2.  Goals
   Adapun tujuan utama dari proyek ini adalah :
-  Meningkatkan efisiensi dan akurasi dalam proses pemilahan sampah
-  Menerapkan pendekatan _transfer learning_ agar model dapat dibangun secara optimal meski dengan data terbatas

3.  Solution Statement
   Solusi yang diterapkan dalam proyek ini guna mencapai tujuan yang telah ditentukan adalah :
-  Penerapan Transfer Learning dengan EfficientNetB3 : menggunakan arsitektur EfficientNetB3 yang telah terbukti efisien dalam hal akurasi dan ukuran model. Serta memanfaatkan bobot pralatih (pretrained weights) dari ImageNet untuk mempercepat pelatihan dan meningkatkan hasil prediksi.
-  Peningkatan performa melalui Hyperparameter tuning : Dengan menyesuaikan parameter penting seperti _learning rate_, jumlah unit dense layer, dropout, dan batch size. Serta, menemukan konfigurasi optimal agar model tidak mengalami overfitting dan dapat bekerja dengan baik pada data yang belum pernah dilihat

Solusi ini akan dievaluasi menggunakan metrik **akurasi** dan **loss** sebagai acuan dalam mengukur keberhasilan model dalam mengklasifikasikan gambar sampah secara otomatis dan akurat.

# Data Understanding
<a href="https://www.kaggle.com/datasets/kaptenyasa/dataset-sampah">Dataset </a> yang digunakan dalam proyek ini merupakan kumpulan citra sampah yang berasal dari laman <a href="https://www.kaggle.com/"> Kaggle </a> yang telah terbagi ke dalam beberapa kategori. Dataset ini bisa diakses dengan tautan sebagai berikut: https://www.kaggle.com/datasets/kaptenyasa/dataset-sampah

Berdasarkan dataset tersebut, kita bisa melihat bahwa data citra yang ada pada dataset tersebut berjumlah 7000 gambar dengan rincian sebagai berikut :

- Organic : 1000 gambar
- Kertas : 1000 gambar
- Plastik : 1000 gambar
- Botol plastik : 1000 gambar
- Kardus : 1000 gambar
- Kaca : 1000 gambar
- Metal 1000 gambar

Berdasarkan eksplorasi awal terhadap dataset citra sampah, dilakukan pengambilan beberapa sampel visualisasi acak untuk data yang tersedia. Gambar-gambar yang ditampilkan dengan rasionya memberikan gambaran visual mengenai karakteristik dari data citra yang ada. Rasio gambar secara umum menunjukkan konsistensi proporsional, sehingga proses_Resize_ atau normalisasi ukuran pada tahap _preprocessing_ selanjutnya dapat dilakukan dengan efisien tanpa mengorbankan informasi penting dari gambar.

## Data Preparation
Pada tahap ini, dilakukan serangkaian proses untuk mempersiapkan data gambar sampah agar siap digunakan dalam proses pelatihan model klasifikasi berbasis deep learning. Proses data preparation dilakukan secara berurutan sebagai berikut :

1.  Memuat dan Mengorganisasi Dataset : Dataset yang digunakan merupakan kumpulan gambar sampah yang sudah dikategorikan ke dalam folder berdasarkan kelasnya.
2.  Resize Gambar : Ukuran asli gambar pada dataset tidak seragam. Oleh karena itu, seluruh gambar diubah ukurannya secara konsisten menjadi 224x224 piksel, agar sesuai dengan arsitektur input model _EfficientNetB1_ yang akan digunakan nantinya. Ini juga memastikan bahwa dimensi input antar gambar seragam selama proses pelatihan
3.  Rescaling (Normalisasi nilai piksel) : Seluruh gambar dinormalisasi dengan cara mengubah nilai piksel dari rentang 0-255 menjadi 0-1. Langkah ini penting untuk mempercepat proses pelatihan model dan mencegah dominasi nilai besar dalam proses komputasi jaringan saraf
4.  Augmentasi Data : Teknik augmentasi gambar diterapkan untuk memperkaya variasi data pelatihan dan mengurangi resiko overfitting./
5.  Splitting Data : Dataset dibagi menjadi tiga subset utama : Data pelatihan, data validasi, dan data testing. Proporsi pemisahan dilakukan sekitar 60:20:20. Hal ini bertujuan agar performa model dapat dievaluasi terhadap data yang belum pernah dilihat selama pelatihan, sehingga memberikan gambaran performa generalisasi model.

## Modelling
Proses pemodelan dilakukan melalui beberapa langkah sebagai berikut : 
1.  Pre-trained Model : Menggunakan **EfficientNetB3** dengan bobot pralatih dari ImageNet dan tanpa bagian fully connected layer pada bagian atas (top layer).
2.  Layer tambahan : Ditambahkan beberapa layer seperti GlobalAveragePooling2D, beberapa Dense layer, dan dropout untuk mengurangi overfitting
3.  Loss Function : **categorical_crossentropy** karena data diklasifikasikan dalam beberapa kelas.
4.  Optimizer : **Adam**, dengan learning rate yang disesuaikan melalui tuning
5.  Metrics : Akurasi digunakan sebagai metrik utama untuk evaluasi performa

Kelebihan dan kekurangan EfficientNetB3
-  Kelebihan :
    -  Efisien secara komputasi : lebih kecil dan lebih cepat dibanding banyak model lain dengan akurasi setara
    -  Performa tinggi pada berbagai tugas klasifikasi gambar, termasuk dataset yang kompleks
    -  Mudah diadaptasi melalui _fine-tuning_ dan kompatibel dengan _transfer learning_
-  Kekurangan :
    -  Memerlukan input gambar dengan resolusi tertentu (ukuran input lebih besar dari model ringan)
    -  Latensi bisa meningkat pada perangkat keras dengan memori rendah karena ukuran model menengah
    -  Tidak selalu optimal tanpa penyesuaian hyperparameter untuk kasus spesifik  
   
## Evaluation
Pada tahap evaluai, dilakukan pengukuran kinerja model menggunakan metrik yang sesuai dengan konteks klasifikasi multi-kelas untuk gambar sampah. Untuk mengevaluasi kinerja model, digunakan dua metrik utama, yaitu :
1.  Accuracy (akurasi)
   Akurasi merupakan metrik yang digunakan untuk mengukur seberapa banyak prediksi model yang benar dibandingkan dengan seluruh prediksi yang dilakukan.
2.  Loss (Categorical Crossentropy Loss)
   Loss function yang digunakan adalah _categorical crossentropy_, yang mengukur seberapa jauh prediksi model dari nilai sebenarnya dalam bentuk probabilitas.

Berdasarkan pelatihan dan validasi model menggunakan _EfficientNetB3_, model menunjukkan performa sebagai berikut : 
-  **Akurasi pada data validasi** mencapai sekitar **0.9014** atau setara dengan **90%**
-  **Loss pada data validasi** menunjukkan nilai yang relatif rendah, yaitu sebesar **0.3180** atau setara dengan **30%**

Penggunaan metrik akurasi dan loss telah memberikan gambaran yang cukup jelas terhadap performa model dalam mengklasifikasikan gambar sampah. Akurasi yang cukup tinggi menunjukkan bahwa model mampu mengenali pola dengan baik, sementara nilai loss yang rendah menandakan bahwa prediksi model cenderung dekat dengan label sebenarnya.
