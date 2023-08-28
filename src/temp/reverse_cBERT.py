
from transformers import RobertaForCausalLM, RobertaTokenizer
import torch.nn as nn
import torch
# Define the model repo
model_name = "seyonec/ChemBERTa-zinc-base-v1"


vector = [0.6031350493431091, 0.6677084565162659, 0.016592182219028473, -0.707610011100769, 0.8135500550270081, 0.43937936425209045, -0.1857025921344757, -0.22041480243206024, -0.30007266998291016, -0.15219171345233917, 0.7677524089813232, -0.28166520595550537, 0.45216479897499084, 0.4764891564846039, -1.0752167701721191, 0.2755366563796997, 0.04008931294083595, 0.8449932932853699, 0.6163045763969421, -0.28297334909439087, -0.1818048357963562, -0.3275540769100189, 0.4180205166339874, 0.593123733997345, -0.6085864901542664, 0.3467053174972534, 0.43668636679649353, 0.18121115863323212, 0.4534023702144623, -0.620210587978363, 0.46210166811943054, -0.3367927372455597, 0.42983290553092957, -0.15430064499378204, -0.3946739733219147, -0.07699890434741974, -0.27268901467323303, -0.44537949562072754, -1.0388494729995728, 0.2532612383365631, -0.7741926908493042, -0.2686239182949066, 0.4159480035305023, 0.26613786816596985, -0.47652697563171387, 1.1492544412612915, 0.23803779482841492, -0.3164541721343994, 0.3345792889595032, 0.3728717565536499, 0.2629034221172333, -0.2670649588108063, 0.8652779459953308, -0.4034826159477234, 0.5934816002845764, 1.1586264371871948, -0.02869415283203125, 0.24004241824150085, 0.49225983023643494, 0.26575762033462524, 0.4071025252342224, 0.9354102611541748, -0.8703439831733704, 0.4459327459335327, -0.08493182808160782, -0.25623977184295654, -0.23348383605480194, -0.3470412790775299, 0.0391787588596344, 0.524294376373291, 0.41177281737327576, -0.6747859716415405, 0.2162427008152008, -0.7705594897270203, 0.31450819969177246, 0.8820747137069702, 0.5823559165000916, 0.3989901840686798, 0.5707885026931763, 0.3600618243217468, -0.30969032645225525, 0.10369537025690079, 0.22013872861862183, 0.6322399377822876, -0.25219061970710754, 0.43031683564186096, -0.7542725801467896, -0.7733712196350098, -0.5192685723304749, 0.17556145787239075, -0.3054167330265045, -0.4345453679561615, 0.34615904092788696, 1.0013240575790405, 0.0953734815120697, 0.647200882434845, -0.14081025123596191, 0.08972985297441483, -0.08112294226884842, 0.25159597396850586, 0.310621440410614, 1.0645300149917603, -0.05440358817577362, 0.2579292356967926, 0.624142050743103, 0.15081170201301575, 0.46660494804382324, 0.32888635993003845, 0.3673418462276459, 0.27384153008461, -0.7837847471237183, -0.7694266438484192, -0.03375035151839256, -0.3015979826450348, -0.20491322875022888, -0.3504980504512787, 0.28464293479919434, 0.1836046725511551, 0.6956000328063965, -0.21369384229183197, -0.09614376723766327, -0.8653160333633423, 0.35096290707588196, 1.0504778623580933, 0.3514708876609802, 0.6774206757545471, -0.2854749858379364, 0.2175336629152298, -0.48701196908950806, 0.1982453614473343, 0.599605143070221, -0.013916140422224998, 0.4786432385444641, 1.0700342655181885, -0.8498259782791138, 0.4981153607368469, 0.7030338048934937, 0.05521747097373009, -0.020655544474720955, -0.4288569986820221, -0.15164123475551605, 0.3480426073074341, -0.5637975335121155, -0.06528102606534958, 0.23880799114704132, -0.50084388256073, 0.8054862022399902, -0.637901782989502, -0.06505950540304184, 0.14426100254058838, 1.138612151145935, -0.27390098571777344, -0.47827112674713135, -0.2911171317100525, 1.0079935789108276, -0.6695325970649719, 0.8747656941413879, -0.10725357383489609, -0.38846439123153687, 1.385069489479065, -0.16526730358600616, -1.2545706033706665, 0.13608984649181366, -0.17291133105754852, 0.7589641213417053, 0.3406135141849518, 0.32538333535194397, 0.7240551114082336, 0.43721240758895874, 0.17420260608196259, 0.11861129105091095, 0.04993065819144249, -1.0274648666381836, 0.18971912562847137, -0.1333567351102829, 0.7962172627449036, 0.006458827294409275, -0.5118764042854309, 0.5158302783966064, 0.6952675580978394, -0.04633874446153641, 0.19223015010356903, -0.6527568101882935, -0.2780943810939789, -0.47638002038002014, -0.9130840301513672, -0.28712037205696106, 0.10155633836984634, -0.2877542972564697, -0.6482436060905457, 0.41858699917793274, -0.29449036717414856, -0.37089788913726807, -0.3745885491371155, -0.25974375009536743, 0.6860666871070862, 0.36809876561164856, 0.17202474176883698, 0.1753399670124054, -0.18426480889320374, -0.9263947010040283, -0.43221142888069153, -0.9288434386253357, -0.5275231599807739, -0.024766569957137108, 0.17683644592761993, 0.36022648215293884, -0.4114159047603607, -0.4214077889919281, -0.7210702300071716, 0.27257683873176575, -0.06985298544168472, 0.18791356682777405, -0.0005546410684473813, 1.1149775981903076, 0.7101526260375977, -0.13233956694602966, -0.6995165348052979, -1.2574125528335571, 0.0005169822834432125, 0.388978511095047, -0.4558910131454468, 1.0783045291900635, -0.555284321308136, -0.25229161977767944, 0.5114899277687073, -0.44880205392837524, 0.2591259479522705, -0.37658289074897766, -0.08655201643705368, -0.08914107829332352, -0.9137884974479675, 0.3057032823562622, 0.003442332847043872, 0.08498741686344147, -0.623676598072052, -0.8772541284561157, -0.5778149962425232, 0.8410254716873169, -0.5941020250320435, -0.36227667331695557, -0.08013919740915298, 0.37641245126724243, 1.0423861742019653, -0.17068441212177277, 0.513220489025116, 0.1684914380311966, 0.056245770305395126, -0.2343466579914093, 0.20889142155647278, -1.2965924739837646, -0.3646620512008667, -0.07115434855222702, -1.116684913635254, 0.35352379083633423, -0.9513688087463379, -0.32858142256736755, 0.08382979780435562, 0.508726954460144, 1.1199792623519897, 0.1198762059211731, -0.776045024394989, 0.6681768894195557, 1.5894304513931274, 0.24476364254951477, -0.2478647530078888, -0.6358031034469604, -0.06363993883132935, -1.1456074714660645, 0.8961095213890076, -0.26398125290870667, 0.5115559101104736, 0.6261292695999146, -0.5633901953697205, 0.29041236639022827, -0.1362462192773819, -0.6686343550682068, 1.0516424179077148, 0.4560888409614563, -0.08229464292526245, 0.16992469131946564, 0.90032559633255, -0.22023069858551025, -0.2618798613548279, 1.423566460609436, 1.1448354721069336, -0.5825109481811523, -0.12498172372579575, -0.8590836524963379, -0.2120833396911621, -0.13620276749134064, -0.08858512341976166, 0.35016754269599915, -0.13857942819595337, 0.678830087184906, -0.5555198788642883, -0.4885840117931366, 0.5054352879524231, -1.5659328699111938, 0.40802210569381714, -0.4828955829143524, 0.02111729048192501, 0.7993718981742859, -0.804084300994873, 0.7680121064186096, -0.3893323540687561, -0.8268929719924927, 0.283893346786499, -0.10108557343482971, -0.13385118544101715, -0.7748317718505859, -0.07447263598442078, 0.11532332003116608, 0.5221795439720154, 0.7774089574813843, -0.7607936263084412, -0.4460621476173401, -0.053000420331954956, 1.1752735376358032, 0.488152414560318, -1.2316350936889648, -0.6730495691299438, 0.17357929050922394, -0.12979573011398315, 0.3815140426158905, 0.2567216753959656, -0.7109089493751526, -0.3153780698776245, 0.2258586436510086, 0.04868104308843613, 0.22923828661441803, 0.11111773550510406, -0.5144873857498169, -1.0228986740112305, -0.4518716037273407, 0.0226674173027277, -0.5306838750839233, 0.06284843385219574, -0.25069165229797363, 0.00552179291844368, 0.049722183495759964, 0.6775991320610046, 0.005999741144478321, 1.201106309890747, 0.05168033763766289, -0.44585347175598145, -0.32586240768432617, -0.21759246289730072, -0.39552950859069824, -0.8317844867706299, 0.6121390461921692, -0.25358739495277405, -0.7372056245803833, -0.4515863060951233, 0.10051361471414566, -1.1747850179672241, -0.362162321805954, 0.0454237274825573, 0.03876181319355965, 0.38596242666244507, 0.4544592499732971, 0.4308032989501953, -0.10824868828058243, -0.3229926526546478, -0.44943204522132874, -1.2448481321334839, 0.48735687136650085, 0.27374547719955444, 0.023798082023859024, 0.30233997106552124, -1.3311119079589844, 0.26317286491394043, -0.5740389227867126, 0.5564273595809937, -0.47928473353385925, -0.2617160379886627, -0.8248160481452942, -0.013435624539852142, 0.6785678863525391, 0.746788740158081, -0.5848591923713684, -0.7988502979278564, -0.03811249881982803, 0.12276885658502579, 0.09474611282348633, -0.6776635050773621, 0.5373656749725342, -0.48679032921791077, -0.6212677955627441, 0.23999452590942383, 0.06004107743501663, -0.28468745946884155, 0.06861554086208344, 0.037071943283081055, 0.401084303855896, -0.2509101331233978, 0.32331088185310364, 0.06522665917873383, -0.4374481737613678, -0.6255251169204712, 0.41828352212905884, 0.2691997289657593, -0.21970227360725403, -0.19463902711868286, -0.3532401919364929, -0.5182662010192871, -0.3690520226955414, -0.7443764805793762, -0.0612671822309494, 0.06969484686851501, -0.8469229340553284, 0.4631824493408203, -0.9718645215034485, 0.5852622985839844, 0.4993266463279724, -0.04920221492648125, 0.4537850320339203, 0.5896273851394653, 0.19734318554401398, 0.4666326940059662, -0.46669039130210876, 0.9244735240936279, 0.23767171800136566, -0.36146074533462524, 0.0316406674683094, -0.19843876361846924, -0.8126562833786011, -0.21081316471099854, -1.3891639709472656, 0.1317022293806076, 0.6902837753295898, 0.4522321820259094, 0.9333395957946777, -0.409695029258728, -0.15363025665283203, 0.5564402341842651, 0.10564736276865005, -0.5882256627082825, 0.5531659722328186, -0.6842247247695923, -0.2552970349788666, -0.09792646020650864, 0.6928629875183105, -0.177494615316391, 0.07522597163915634, 0.17423981428146362, -0.02454754337668419, -0.3796316087245941, 0.8011660575866699, -0.005184166599065065, 0.8984843492507935, 0.6522912979125977, -1.1264283657073975, 0.9246616363525391, 0.6904774308204651, -0.8887795209884644, -0.394095242023468, -1.0110105276107788, -0.6171382069587708, 0.5035606622695923, 0.11005020141601562, 0.1453750729560852, -0.2110345959663391, 0.46734619140625, -0.507094144821167, -0.7477138042449951, -0.7771638035774231, -0.12519770860671997, 0.4155725836753845, -0.07858218997716904, -0.684346616268158, 0.3000658452510834, -0.263627290725708, -1.037580966949463, -0.5303855538368225, 0.013224647380411625, -0.3160267472267151, -0.46737751364707947, -0.31476902961730957, -0.21923311054706573, -0.3277336657047272, 0.6002251505851746, 0.001673539518378675, -0.16389200091362, -0.7844948768615723, -0.2464725375175476, -0.16174796223640442, -0.3790018558502197, 0.3461766541004181, 0.050750233232975006, 0.35178858041763306, 0.8556419610977173, -0.2696067988872528, 0.3021322786808014, 0.1337697058916092, -1.0337330102920532, 0.014070465229451656, -0.00922758225351572, -0.40654805302619934, 1.2656867504119873, 0.5867262482643127, 0.3810175955295563, -0.800458550453186, -0.8448618650436401, 0.8635786771774292, 0.5591644644737244, -1.1699740886688232, 0.18790511786937714, -0.3872202932834625, 0.5779711604118347, -1.0274392366409302, 0.1706511676311493, -0.8128314018249512, -0.9490840435028076, 0.3320055603981018, -1.2506288290023804, 0.5840698480606079, 0.9915289282798767, -0.5384981036186218, -0.7451462745666504, 0.7320595979690552, -0.5142161846160889, -1.0543053150177002, -0.409274160861969, 0.5300614237785339, -0.22054961323738098, 0.4812067449092865, 0.46316173672676086, 0.2921885848045349, -0.505398154258728, -0.7310932874679565, -0.10650894790887833, -0.43478602170944214, 0.09651400148868561, -0.45820558071136475, -0.18803264200687408, 0.02506212331354618, -0.7700465321540833, 0.04351251199841499, -0.3623875081539154, -0.6255286335945129, -0.147924542427063, -0.20660516619682312, -0.084401436150074, 0.8258814215660095, -0.8351455926895142, 0.6259046196937561, -0.8869398236274719, 0.7202050089836121, 1.1849628686904907, -0.2441941201686859, 0.7279065251350403, -0.01857457309961319, -0.25302019715309143, 0.5457435846328735, -0.7022278904914856, -0.2464102953672409, -0.11084599792957306, 0.4688548445701599, 0.19823342561721802, -0.28101059794425964, 0.5543763637542725, 0.5244501233100891, 0.2915172576904297, -0.3465937077999115, 0.4522501230239868, -0.05452834442257881, -0.9976690411567688, 0.250122606754303, 1.1989531517028809, 0.24860379099845886, -0.9354918599128723, -0.18407422304153442, 0.4616773724555969, -0.37911009788513184, 0.23075354099273682, 0.486477255821228, -1.1391639709472656, -0.4352167248725891, -0.3424247205257416, 1.7618783712387085, -0.08792638778686523, -0.8749797344207764, 0.8390984535217285, 1.1024590730667114, -0.6158193945884705, 0.3429665267467499, -0.5823107957839966, -0.41595035791397095, -0.6411274671554565, -0.9477682709693909, -0.5807393193244934, 0.4004453122615814, -0.18399211764335632, -0.3112417459487915, 0.700285792350769, -0.6760048270225525, -0.48363786935806274, 0.2640659511089325, 0.276919960975647, 0.483822762966156, -0.13817358016967773, 0.584825336933136, 0.5916083455085754, -0.7682968974113464, 0.33207279443740845, -0.556360125541687, 0.3825661838054657, 0.06158626824617386, 0.24790509045124054, -1.304114580154419, -1.0838911533355713, -0.24820515513420105, -0.18177227675914764, -0.1786835938692093, -0.5346583724021912, -0.5251923203468323, 0.028162451460957527, 0.4195899963378906, -0.19052816927433014, 0.00352446804754436, 0.6894568204879761, 0.5493638515472412, 1.002550721168518, 0.32827651500701904, 0.35872575640678406, 0.1510508507490158, 0.7830429077148438, -0.7244753241539001, 0.5939440131187439, 0.6468254923820496, -0.38838595151901245, 0.5096514225006104, 0.2080775499343872, 1.0620551109313965, 0.2701149582862854, 0.2809862494468689, 0.21228213608264923, 0.8285564184188843, -0.4022800922393799, -0.7526264190673828, 0.044618140906095505, 0.5731393098831177, 0.6317240595817566, 0.08168873190879822, 0.033030182123184204, 1.0236363410949707, 0.1487603634595871, 0.4458615779876709, -0.5636072754859924, 0.7809505462646484, 0.6141932606697083, -0.22794201970100403, 0.0702725276350975, 0.04582434520125389, 0.11752444505691528, -0.3263067603111267, 0.005501735955476761, 0.012666520662605762, -0.7936519384384155, 0.1443186104297638, -0.16169044375419617, 0.2837793231010437, 0.6637398600578308, -0.4183408319950104, -0.21325229108333588, 0.08776164799928665, 0.064314104616642, 1.293723702430725, 0.7521762847900391, -0.3347955346107483, -0.09765928238630295, -0.3748653829097748, -1.1455148458480835, -0.536906361579895, -0.9519743919372559, -0.882536768913269, -0.9492931365966797, 0.6829972267150879, 0.8390374779701233, 1.2767254114151, 0.033156782388687134, -0.12141014635562897, -0.21527224779129028, -0.10358880460262299, 0.7303763628005981, -0.30236876010894775, 0.8424911499023438, -0.5719189643859863, 0.47659438848495483, 0.549950361251831, -0.48317691683769226, -0.24495337903499603, -0.025598378852009773, -0.18036004900932312, 0.1337369680404663, 1.152117371559143, 0.5624839067459106, 0.430996298789978, 1.367914080619812, -0.5339687466621399, 0.040916625410318375, 0.08235104382038116, 0.08994144201278687, 0.4159739911556244, 0.5351424217224121, 0.3795135021209717, 0.5132085680961609, -0.2557748556137085, -0.020131826400756836, -1.0297435522079468, 1.5540192127227783, -0.48433440923690796, 0.740313708782196, 0.02832491137087345, -0.19657516479492188, -0.23065657913684845, -0.9879072308540344, 0.01971232332289219, 1.3548146486282349, -0.6170234680175781, 0.4140721261501312, -0.015198900364339352, -1.0488965511322021, 0.04838130995631218, 0.05418770760297775, -0.57822185754776, 0.5751208066940308, -0.2840220034122467, -0.02656419575214386, -0.14890582859516144, 0.5511435270309448, -0.3183189630508423, 0.47760632634162903, -0.12539103627204895, -0.11062698066234589, -0.314971923828125, 0.25094908475875854, 0.01813622936606407, -0.5297362804412842, 0.7415908575057983, -0.20214608311653137, -1.0254778861999512, -0.3880697786808014, 0.5698779225349426, -0.6478468775749207, -0.3522750437259674, 0.303747296333313, -0.4104920029640198, 1.30471670627594, -0.12837478518486023, -0.16260597109794617, -0.9077590703964233, 0.7520726323127747, -0.4661036431789398, 0.314565509557724, 0.24414288997650146, 0.38455235958099365, -0.22052283585071564, 0.28422123193740845, -0.7662646174430847, 0.06922826915979385, -0.7799861431121826, 1.2242767810821533, 0.010647380724549294, -0.2082650363445282, -0.027688968926668167, 0.845970630645752]

# Download pytorch model
model = RobertaForCausalLM.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

# Generate text using the model
generated_ids = model.generate(vector, max_length=50, pad_token_id=tokenizer.pad_token_id)

# Convert the generated IDs back to text using the tokenizer
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)

