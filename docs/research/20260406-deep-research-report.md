# カリンバ自動採譜に有用なアルゴリズムの横断的サーベイ

## 研究目的と問題設定

録音データから「何の音が、いつ鳴ったか（音高・オンセット・持続・同時発音）」を推定して記譜（MIDI/楽譜相当）へ落とす問題は、一般に **Automatic Music Transcription（AMT）** と呼ばれ、(multi-)pitch 推定、オンセット/オフセット検出、音源（楽器）識別、拍や小節などの構造推定まで複数サブタスクの組み合わせとして整理されます。citeturn15search0

ご相談はこのうち、とくに **録音からカリンバのノート（主に各 tine の発音イベント）を取り出す**工程に焦点があり、ユーザー側の分析で整理された要件（トーナル/ノイズ判別、オンセット検出、ポリフォニー、録音条件不変性、将来のストリーミング適合、ラメラフォン固有性）を満たしやすい研究領域・アルゴリズムを、MIR/音声信号処理/楽器音響/機械学習まで広く横断して位置づけます。citeturn15search0turn25view0

本レポートでは、カリンバ（mbira/lamellophone）が **金属 tine を指で弾いて発音する撥弦型の有音高打楽器（plucked idiophone）**であるという前提（「鋭いアタック＋減衰サステイン」「倍音が豊か」「固定ピッチ」「共鳴・重なり」など）を、既存研究で取り扱われてきた近縁課題（ピアノ、木管合奏、打楽器の音高推定、マルチ F0 推定、ソース分離等）へ写像しながら、実装可能な候補群を抽出します。citeturn28search7turn15search0

## カリンバの信号モデルが示唆する設計上のポイント

### 物理モデル由来の「倍音の非整数比」

カリンバの各 tine（鍵）のスペクトルは、基音 f(1) と支配的な倍音 f(2), f(3) の関係が **矩形断面梁の横振動モード**として説明され、倍音列は **非調和（inharmonic）**で、比 f(2)/f(1), f(3)/f(1) が整数にならないこと、さらに支持点（ブリッジ）が波形を分割するため **比が固定値ではなく支持位置に依存**することが報告されています。ここは、ギター/声のような「ほぼ整数倍の倍音」を暗黙仮定する古典的 harmonicity 指標や comb sum を、そのまま使うと破綻しやすい理由になります。citeturn28search13

この「非調和倍音」は、マルチ F0 推定で広く使われる **harmonicity/スペクトル平滑性**系の推定器でも重要論点として扱われており、たとえば multipitch 推定で **inharmonicity factor を推定しながら頑健化**する設計が古くから議論されています。citeturn10view2turn19view1

### 「共鳴・残響」と「新規オンセット」の区別は、楽器モデリング問題に近い

ユーザーの現状課題（sympathetic resonance による spurious 二次成分、残響中の新規オンセット検出）は、ピアノやハープ等の弦楽器で研究されてきた **sympathetic vibration（共鳴弦振動）**の扱いと構造が似ています。弦楽器側の文献では、響板（beam/plate）と多数弦の結合系としてモードを求め、どのモードが共鳴を支配するかを解析する枠組みが提示されています。citeturn17view0

また AMT の文脈でも、共鳴を「不要成分」ではなく「現象として測って判定する」例があり、たとえばピアノのペダル起点検出で、単音成分を引き算して残差（residual）を取り、残差エネルギー等から sympathetic resonance を測る設計が提案されています。これは、「まず主成分（確からしいノート）を説明し、残りを見て共鳴・ノイズ・境界アーチファクトを判別する」という分析順序を示唆します。citeturn6view2

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["kalimba tines close up","kalimba thumb piano playing close up","spectrogram plucked instrument note","harmonic percussive separation spectrogram example"],"num_per_query":1}

### パーカッション楽器研究の知見は「減衰」「モード」「放射」にある

打楽器の音響研究では、振動モードの理論・実験計測（モーダルテスト）・音放射・物理モデリングが体系的に整理されており、「楽器固有のモード構造と減衰の扱い」が、特徴量設計やテンプレート設計（attack/sustain/decay）に直結します。citeturn6view1

## 研究領域を横断するための「探索地図」

カリンバ採譜に直結しやすい研究領域は、AMT の整理に沿うと概ね次の束に分かれます（括弧内は代表的なキーワード群）。citeturn15search0

第一に、**時間周波数表現と前処理**（STFT/CQT/HCQT、resonator filterbank、HPSS、ノイズ抑圧）。CQT は音楽的に等比（対数）周波数スケールを持つため、音楽信号の特徴抽出として古典的です。citeturn12search0turn12search23

第二に、**トーナル/ノイズ判別**（spectral flatness、entropy flatness、harmonic ratio、periodicity）。スペクトル平坦度は「幾何平均/算術平均」で定義され、スケール不変（ゲインに対して不変）である点が、録音距離変化への頑健性と相性が良い一方、ゼロ成分に敏感などの落とし穴も整理されています。citeturn21view0turn8view3

第三に、**オンセット検出**（spectral flux、位相偏差、complex-domain、検出関数の融合）。オンセット検出は MIR の中核テーマで、スペクトルフラックス、位相、複素領域を用いる系列が比較・改良されています。citeturn8view0turn8view1turn2search0

第四に、**マルチ F0 推定・ノート追跡**（iterative cancellation、spectral smoothness、inharmonicity 対応、NMF/PLCA、HMM/CRF）。倍音重なり（例：オクターブ関係）をどう抑えるか、時間的にどうノートへ束ねるかが主戦場です。citeturn19view1turn19view0turn10view2

第五に、**評価法とベンチマーク**（MIREX、mir_eval、frame-wise vs note-wise）。MIREX では multi-F0 推定/追跡の体系的評価が行われており、評価手続き（許容誤差、マッチング方式）が実装に影響します。citeturn27search0turn27search3turn27search1

探索クエリの例としては、ユーザーが挙げた候補に加え、(a) “inharmonic multipitch estimation”, “spectral smoothness”, “octave errors”, (b) “attack sustain decay templates HMM transcription”, (c) “constant-Q resonator real-time multipitch”, (d) “polyphonic pitch detection streaming ConvLSTM”, (e) “harmonic + inharmonic NMF pitch transcription” あたりを軸にすると、カリンバ固有問題（非調和倍音と共鳴）に近づきます。citeturn19view1turn20view0turn27academia40

## 有望な古典的DSP・確率モデル系アプローチ

### 対数周波数表現：CQT/共振器バンクは「固定ピッチ・少数音」を活かしやすい

CQT は中心周波数が幾何級数で配置されるため、音楽音高の尺度に合いやすく、AMT 入力として繰り返し採用されています。citeturn12search0turn23view2  
計算面では、CQT を効率よく計算して（近似）逆変換まで扱う実装指針が提示されており、WebAudio/WASM を見据える場合にも参考になります。citeturn12search23

一方、リアルタイム志向では、**共振器（resonator）型の time-frequency 表現**を用いた polyphonic pitch 推定が提案されており、計算量と性能のトレードオフを明示しています。たとえば定数Qの resonator を用いる実装で、当時の標準PCで「実時間より速い」ことを目標にした設計が報告されています。citeturn10view1  
カリンバは音域・音集合が小さい（ユーザー前提では 17 音）ため、CQT/共振器バンク上で **「候補周波数が有限」**という制約を強く使えるのが利点です。citeturn20view0turn10view1

### 前処理：HPSSは「オンセット用」と「音高用」を分ける発想に使える

Harmonic/Percussive Source Separation（HPSS）は、スペクトログラム上で「調波＝水平リッジ」「打撃＝垂直リッジ」という近似に基づき、中央値フィルタ等でマスクを作る手法が古典的で高速です。citeturn5view1turn22view0  
ただし、撥弦金属 tine の **アタックは“percussiveっぽい”**ため、HPSS をそのまま適用すると「本来同一ノートのアタック成分が percussive 側へ回る」副作用が起こり得ます（中央値フィルタ型HPSSが“アタックを打楽器として拾う”傾向は論文中で注意されています）。citeturn5view1  
この性質は逆に、**オンセット検出は percussive 側、ピッチ推定は harmonic 側**という二系統特徴量に分ける設計として利用できます。citeturn22view0turn8view0

### トーナル/ノイズ判別：spectral flatness を「帯域分割」して使うのが現実的

ユーザーが課題として挙げた「BW90 や centroid では genuine note も広帯域に散る」は、カリンバのように倍音が豊か（しかも非調和）な信号で起きやすい典型現象です。citeturn28search13turn21view0  
この状況で spectral flatness は、(a) 幾何平均/算術平均比として定義されスケール不変、(b) “白色雑音で 1、純音で 0”という基礎性質を持つため、絶対ゲインに依存しにくい「構造の有無」判定器として位置づけられます。citeturn8view3turn21view0

ただし古典的 flatness はゼロ近傍値に極端に弱く、安定化のためにエントロピーに基づくロバスト化（あるいは entropy spectral flatness）が提案されています。citeturn21view0turn8view3  
さらに、全帯域で“平坦”を測ると「狭帯域にエネルギーが集中する打撃音（例：バスドラム）を誤ってトーナル扱いする」問題が指摘され、**周波数セグメントごとの flatness**を使う改良が提案されています。カリンバの「倍音で広帯域」問題に対しても、帯域セグメント化は理にかないます。citeturn8view3

### オンセット検出：複数検出関数の融合と、位相・複素領域の利用

オンセット検出では、スペクトルフラックス（振幅差分）だけでなく、位相偏差や複素領域差分（magnitudeとphaseを同時に扱う）を用いることで、鋭い検出関数を作る系譜があります。citeturn8view0turn8view1turn2search0  
複素領域のオンセット検出は、「エネルギー系＋位相系」の両方の情報を扱い、低誤検出率で高検出率を狙う方向性として提案されています。citeturn8view1turn8view0

また、検出器の融合（複数の検出関数を組み合わせる）を体系立てて比較・評価する研究もあり、単一特徴量が録音条件に弱い場合に「融合でロバスト化する」という方針を後押しします。citeturn27search14

### マルチF0推定：harmonicity＋smoothness＋inharmonicity 対応が「倍音重なり」に効く

マルチF0推定の古典では、最も目立つ F0 を推定してスペクトルから引き算し、残差に対して繰り返す **iterative cancellation**が中心的アイデアとして整理されています。citeturn10view1turn10view2  
この系列では、inharmonicity を考慮するために「倍音位置のずれ」を許容し、inharmonicity factor を推定して推定を整合させる設計が示されています。カリンバが梁振動由来の非調和倍音を持つ点と整合します。citeturn10view2turn28search13

ピアノ向けの multipitch 推定では、倍音重なり（とくにオクターブ関連）を主問題として、**スペクトル包絡を平滑（ARモデル）として推定**し、誤った sub-octave や harmonic/subharmonic エラーを尤度で落とす枠組みが提案されています。ここでは flatness（白色化の度合い）を目的関数に含めており、ユーザーが注目している flatness を「誤推定抑圧の理論的部品」として組み込めることを示唆します。citeturn19view1  
同様に、log周波数上の salience と “harmonically-related F0 の誤検出抑圧” を明示したピアノ向け multi-F0 推定もあり、倍音重なりを「ルール＋統計」で抑える設計が整理されています。citeturn13view0turn19view1

### 因子分解：固定音集合なら「固定辞書」NMF/PLCA が強い

カリンバは「対象音が少ない」「個体差はあるが各 tine のピッチは固定」という点で、汎用AMTより **“特定楽器・固定辞書”**が作りやすい部類です。この設定は、(a) 事前収録した単音から辞書 W を作り、(b) 係数 H を推定する **非負行列分割（non-negative matrix division）**として定式化する提案と相性が良いです。citeturn20view0

さらに、音の時間発展（attack/sustain/decay）をテンプレート状態として持ち、状態順序を HMM 制約で制御する shift-invariant PLCA 系は、「新規オンセット」と「残響/共鳴中の成分」を状態遷移として表現でき、カリンバのアタック・減衰モデル化に直結します。citeturn19view0turn15search0

## 深層学習系アプローチとストリーミング適合

### 「オンセットを別目的で学習」する設計は、撥弦・打撃系の本質に合う

深層学習によるピアノ採譜の代表例では、オンセットとフレーム（音高活性）を同時に学習する **dual-objective**設計が採用され、オンセットを強く使って note event を安定化させています。citeturn23view2  
この系統を詳細分析した研究では、（少なくともピアノ条件で）**オンセットが最重要の注意対象**であり、さらに **ルールベースの後処理（オンセットでフレームをゲートする）**が note-wise 精度を大きく左右する、という観察が報告されています。カリンバでも「新規アタック vs 共鳴」を区別する必要があるため、オンセット強化＋後処理強化は移植価値が高いです。citeturn25view0

### 「軽量・汎用」モデル群：単一楽器のポリフォニー採譜に転用しやすい

近年は、低リソース端末でも動くことを前提に、onset/multipitch/note activation をマルチ出力で予測する **軽量な instrument-agnostic AMT**モデルが提案されています。これは「専用モデルは高精度だが、実運用ではモデル数・計算制約が障壁」という問題意識から設計されています。citeturn23view0  
同系統で、実用ツールとして公開されている **Basic Pitch** は「instrument-agnostic」「polyphonic」「単一楽器の転写で良好」という位置づけを明示しており、カリンバ単独録音に対するベースラインとして有力です。citeturn3search29  
（この種のモデルは、カリンバ固有の非調和倍音・共鳴に最適化されていない可能性があるため、後述の“特定辞書/特定物理”戦略で補強するのが現実路線です。citeturn15search0turn23view0）

### マルチ楽器・大規模化：Transformer系（MT3など）は「表現力」は強いが重い

AMTを「イベント列（トークン列）予測」として扱う Transformer 系（例：MT3）は、複数データセット/複数楽器を統合して学習し、従来の“楽器別モデル”の壁を越える方向性を提示しています。citeturn23view3turn24view0  
一方で、ストリーミング/低遅延や Web 実装（WASM）を強く意識する場合、Transformer は計算コストとコンテキスト（窓長）設計がボトルネックになりやすく、軽量CNN/ConvRNNや古典DSPとのハイブリッドの方が現実的になりやすい、という設計含意は残ります。citeturn23view0turn27academia40

### ストリーミング前提の研究：オンライン polyphonic pitch detection

「ストリーミングで音声→MIDI」のオンライン設計を明示した研究として、ConvLSTM ベースで音声を逐次入力しつつ polyphonic pitch を推定するシステムが報告されています。これはユーザー要件 E（causal/低遅延）に直接対応する探索領域です。citeturn27academia40  
また、階層的に f0→オンセット/オフセット→ノート修正へ進む深層学習パイプライン（deep layered learning）も提案されており、「まず音高の“地図”を作り、あとからノートとして整形する」順序が、共鳴・重なり・誤検出除去に効くことを示唆します。citeturn27academia41

加えて、ラベル不足を前提に自己教師ありで multi-pitch を学ぼうとする研究（transposition equivariance など）も出てきており、「カリンバ固有データが少ない」を補う方向性として注視に値します。citeturn1search26

## カリンバ採譜への落とし込み設計

ここでは、ユーザーが挙げた要件 A〜F を満たしやすい“実装の型”を、研究知見に基づいて2系統（モデルベース/学習ベース）に整理します。

### モデルベース寄り：固定辞書＋オンセット強化＋状態モデルで「共鳴と誤検出」を抑える

**固定辞書（各 tine の音色テンプレ）**を使ってスペクトログラムを説明し、係数系列をノートへ整形する設計は、カリンバの「少数固定音・固定ピッチ」を最大限に活かします。まず単音サンプルから辞書を作り、非負行列分割で係数（活性）を推定する枠組みが提案されています。citeturn20view0  
辞書を「attack/sustain/decay の複数状態テンプレ」に拡張し、HMMで状態順序を制約する shift-invariant PLCA 系は、**“新規 pluck なら attack 状態を通る”**という構造を持てるため、sympathetic resonance のような「アタックらしさの薄い立ち上がり」を、状態制約で弾きやすくします。citeturn19view0turn25view0

このとき時間周波数表現は、(a) 音楽スケール整合の CQT、(b) 高速化・ストリーミングを見据えた共振器バンク、のどちらでも設計可能で、CQTの理論は古典文献で定義され、実装指針（高速計算）も提示されています。citeturn12search0turn12search23turn10view1  
さらにカリンバ固有の非調和倍音は、単純な“整数倍 comb”ではなく、個体ごとの倍音位置・比を前提にしたテンプレ化が必要であり、その物理的起源（梁モード・支持位置依存）はカリンバ分析論文が明示しています。citeturn28search13

**トーナル/ノイズ判別**については、ゲイン不変性が重要なら spectral flatness を軸にしつつ、ゼロ値感度の問題を避けるエントロピー系 flatness や、帯域セグメント化（segmental flatness）を併用する設計が、文献上の弱点整理と整合します。citeturn21view0turn8view3  
**オンセット検出**は、スペクトルフラックス単独よりも、位相・複素領域差分を含む検出関数を候補に置き、条件により融合する方針が研究史的に支持されます。citeturn8view0turn8view1turn27search14

最後に **倍音重なり（例：オクターブ関係）**は、ピアノ系 multipitch 研究で主問題として扱われ、スペクトル平滑性（AR包絡）や尤度最大化で sub-octave/harmonic confusion を落とす設計が具体的です。カリンバでも「重なりの構造は違うが、誤りの型は似る」ため、この種の“誤り抑圧機構”を部品として移植する価値があります。citeturn19view1turn10view2

### 学習ベース寄り：軽量AMTモデルをベースラインにして「オンセット＋後処理」で寄せる

学習ベースの近道は、(a) instrument-agnostic の軽量モデル（例：Basic Pitch相当）をベースラインにして、(b) カリンバ単音サンプルや合成データで微調整/後処理を作る流れです。Basic Pitch は polyphonic・instrument-agnostic を掲げ、単一楽器の転写で良いという前提を明にしています。citeturn3search29  
軽量な instrument-agnostic AMT を設計する研究でも、onset/multipitch/note activation のマルチ出力が有効で、低メモリ/低計算を意識した構成が提案されています。citeturn23view0

また、オンセットとフレームを同時に学習し、オンセットでフレームをゲートする“Onsets and Frames 系”設計は、モデル解析研究により **後処理（rule-based inference）が note-wise 指標を大きく左右**することが示されており、カリンバでも「オンセット検出器＋後処理器」を丁寧に作る方が効く可能性が高いです。citeturn25view0turn23view2  
ストリーミング要件が強い場合は、オンライン polyphonic pitch detection を目標にした ConvLSTM 系の研究が「逐次入力→MIDI」を明示しており、モデル構造・遅延設計の参照点になります。citeturn27academia40

### 評価設計：frame-wise と note-wise を分け、許容誤差を固定する

AMT の評価は、frame-wise（各フレームで正しい音高集合か）と、note-wise（オンセット/オフセットを含むノートイベントとして正しいか）で性質が大きく異なり、note-wise ではオンセット・オフセット許容（例：オンセット±50ms、オフセットは max(50ms, 20% duration)）が一般的に用いられます。citeturn27search25turn27search7  
一方で MIREX と mir_eval ではマッチングの実装差（greedy vs bipartite matching など）があることが明記されており、研究比較や回帰テストでは「どの実装で測ったか」を固定する必要があります。citeturn27search1turn27search2  
multi-F0 推定/追跡の体系的評価は MIREX 由来の議論があり、指標設計の参考になります。citeturn27search0turn27search3

### 実装上の補足：ハードウェア/収録系で難易度が大きく変わる

アルゴリズムと独立ではありませんが、lamellophone 系の研究開発では、ボディ下面に piezo pickup を付けて「外乱や空気伝搬の混入を減らし、堅牢な信号を得る」設計が採られており、録音条件不変性（要件D）を“信号入力側”から底上げする選択肢になります。citeturn28search2