ГС
═г
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.12v2.3.0-54-gfcc4b966f18Вп
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	dн*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	dн*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:н*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:н*
dtype0
~
covtr2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecovtr2/kernel
w
!covtr2/kernel/Read/ReadVariableOpReadVariableOpcovtr2/kernel*&
_output_shapes
: *
dtype0
n
covtr2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecovtr2/bias
g
covtr2/bias/Read/ReadVariableOpReadVariableOpcovtr2/bias*
_output_shapes
: *
dtype0
К
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namebatch_normalization/gamma
Г
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
: *
dtype0
И
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namebatch_normalization/beta
Б
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0
Ц
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean
П
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0
Ю
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance
Ч
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
: *
dtype0
~
covtr3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *
shared_namecovtr3/kernel
w
!covtr3/kernel/Read/ReadVariableOpReadVariableOpcovtr3/kernel*&
_output_shapes
:@ *
dtype0
n
covtr3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namecovtr3/bias
g
covtr3/bias/Read/ReadVariableOpReadVariableOpcovtr3/bias*
_output_shapes
:@*
dtype0
О
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma
З
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta
Е
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean
У
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
в
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance
Ы
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0

covtr4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А@*
shared_namecovtr4/kernel
x
!covtr4/kernel/Read/ReadVariableOpReadVariableOpcovtr4/kernel*'
_output_shapes
:А@*
dtype0
o
covtr4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namecovtr4/bias
h
covtr4/bias/Read/ReadVariableOpReadVariableOpcovtr4/bias*
_output_shapes	
:А*
dtype0
П
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_2/gamma
И
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:А*
dtype0
Н
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_2/beta
Ж
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_2/moving_mean
Ф
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:А*
dtype0
г
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_2/moving_variance
Ь
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:А*
dtype0
{
cov3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namecov3/kernel
t
cov3/kernel/Read/ReadVariableOpReadVariableOpcov3/kernel*'
_output_shapes
:А*
dtype0
j
	cov3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	cov3/bias
c
cov3/bias/Read/ReadVariableOpReadVariableOp	cov3/bias*
_output_shapes
:*
dtype0

NoOpNoOp
ь@
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*з@
valueЭ@BЪ@ BУ@
И
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
R
&	variables
'trainable_variables
(regularization_losses
)	keras_api
Ч
*axis
	+gamma
,beta
-moving_mean
.moving_variance
/	variables
0trainable_variables
1regularization_losses
2	keras_api
R
3	variables
4trainable_variables
5regularization_losses
6	keras_api
h

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
R
=	variables
>trainable_variables
?regularization_losses
@	keras_api
Ч
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
R
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
h

Nkernel
Obias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
R
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
Ч
Xaxis
	Ygamma
Zbeta
[moving_mean
\moving_variance
]	variables
^trainable_variables
_regularization_losses
`	keras_api
R
a	variables
btrainable_variables
cregularization_losses
d	keras_api
h

ekernel
fbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
ж
0
1
 2
!3
+4
,5
-6
.7
78
89
B10
C11
D12
E13
N14
O15
Y16
Z17
[18
\19
e20
f21
v
0
1
 2
!3
+4
,5
76
87
B8
C9
N10
O11
Y12
Z13
e14
f15
 
н
	variables
klayer_metrics

llayers
trainable_variables
mnon_trainable_variables
nlayer_regularization_losses
ometrics
regularization_losses
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
н
	variables
player_metrics

qlayers
trainable_variables
rnon_trainable_variables
slayer_regularization_losses
tmetrics
regularization_losses
 
 
 
н
	variables
ulayer_metrics

vlayers
trainable_variables
wnon_trainable_variables
xlayer_regularization_losses
ymetrics
regularization_losses
YW
VARIABLE_VALUEcovtr2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
н
"	variables
zlayer_metrics

{layers
#trainable_variables
|non_trainable_variables
}layer_regularization_losses
~metrics
$regularization_losses
 
 
 
▒
&	variables
layer_metrics
Аlayers
'trainable_variables
Бnon_trainable_variables
 Вlayer_regularization_losses
Гmetrics
(regularization_losses
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

+0
,1
-2
.3

+0
,1
 
▓
/	variables
Дlayer_metrics
Еlayers
0trainable_variables
Жnon_trainable_variables
 Зlayer_regularization_losses
Иmetrics
1regularization_losses
 
 
 
▓
3	variables
Йlayer_metrics
Кlayers
4trainable_variables
Лnon_trainable_variables
 Мlayer_regularization_losses
Нmetrics
5regularization_losses
YW
VARIABLE_VALUEcovtr3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

70
81

70
81
 
▓
9	variables
Оlayer_metrics
Пlayers
:trainable_variables
Рnon_trainable_variables
 Сlayer_regularization_losses
Тmetrics
;regularization_losses
 
 
 
▓
=	variables
Уlayer_metrics
Фlayers
>trainable_variables
Хnon_trainable_variables
 Цlayer_regularization_losses
Чmetrics
?regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
D2
E3

B0
C1
 
▓
F	variables
Шlayer_metrics
Щlayers
Gtrainable_variables
Ъnon_trainable_variables
 Ыlayer_regularization_losses
Ьmetrics
Hregularization_losses
 
 
 
▓
J	variables
Эlayer_metrics
Юlayers
Ktrainable_variables
Яnon_trainable_variables
 аlayer_regularization_losses
бmetrics
Lregularization_losses
YW
VARIABLE_VALUEcovtr4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

N0
O1

N0
O1
 
▓
P	variables
вlayer_metrics
гlayers
Qtrainable_variables
дnon_trainable_variables
 еlayer_regularization_losses
жmetrics
Rregularization_losses
 
 
 
▓
T	variables
зlayer_metrics
иlayers
Utrainable_variables
йnon_trainable_variables
 кlayer_regularization_losses
лmetrics
Vregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1
[2
\3

Y0
Z1
 
▓
]	variables
мlayer_metrics
нlayers
^trainable_variables
оnon_trainable_variables
 пlayer_regularization_losses
░metrics
_regularization_losses
 
 
 
▓
a	variables
▒layer_metrics
▓layers
btrainable_variables
│non_trainable_variables
 ┤layer_regularization_losses
╡metrics
cregularization_losses
WU
VARIABLE_VALUEcov3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	cov3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

e0
f1

e0
f1
 
▓
g	variables
╢layer_metrics
╖layers
htrainable_variables
╕non_trainable_variables
 ╣layer_regularization_losses
║metrics
iregularization_losses
 
v
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
*
-0
.1
D2
E3
[4
\5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

-0
.1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

D0
E1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

[0
\1
 
 
 
 
 
 
 
 
 
 
 
 
|
serving_default_gen_noisePlaceholder*'
_output_shapes
:         d*
dtype0*
shape:         d
ї
StatefulPartitionedCallStatefulPartitionedCallserving_default_gen_noisedense/kernel
dense/biascovtr2/kernelcovtr2/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancecovtr3/kernelcovtr3/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancecovtr4/kernelcovtr4/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancecov3/kernel	cov3/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         mY*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *.
f)R'
%__inference_signature_wrapper_1172719
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ў	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp!covtr2/kernel/Read/ReadVariableOpcovtr2/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp!covtr3/kernel/Read/ReadVariableOpcovtr3/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp!covtr4/kernel/Read/ReadVariableOpcovtr4/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOpcov3/kernel/Read/ReadVariableOpcov3/bias/Read/ReadVariableOpConst*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *)
f$R"
 __inference__traced_save_1173567
╣
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biascovtr2/kernelcovtr2/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancecovtr3/kernelcovtr3/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancecovtr4/kernelcovtr4/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancecov3/kernel	cov3/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *,
f'R%
#__inference__traced_restore_1173643вз
Ъ
Л
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1171864

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @:::::i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
▐
e
F__inference_dropout_2_layer_call_and_return_conditional_losses_1172356

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/ConstО
dropout/MulMulinputsdropout/Const:output:0*
T0*B
_output_shapes0
.:,                           А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╧
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*B
_output_shapes0
.:,                           А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┘
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*B
_output_shapes0
.:,                           А2
dropout/GreaterEqualЪ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*B
_output_shapes0
.:,                           А2
dropout/CastХ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*B
_output_shapes0
.:,                           А2
dropout/Mul_1А
IdentityIdentitydropout/Mul_1:z:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           А:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
м
f
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_1172218

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           @2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
дЬ
И

F__inference_Generator_layer_call_and_return_conditional_losses_1173038

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource3
/covtr2_conv2d_transpose_readvariableop_resource*
&covtr2_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource3
/covtr3_conv2d_transpose_readvariableop_resource*
&covtr3_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource3
/covtr4_conv2d_transpose_readvariableop_resource*
&covtr4_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource1
-cov3_conv2d_transpose_readvariableop_resource(
$cov3_biasadd_readvariableop_resource
identityИа
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	dн*
dtype02
dense/MatMul/ReadVariableOpЖ
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         н2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:н*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         н2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:         н2

dense/Reluf
reshape/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
reshape/ShapeД
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackИ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1И
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2Т
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3ъ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeб
reshape/ReshapeReshapedense/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:         2
reshape/Reshaped
covtr2/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
covtr2/ShapeВ
covtr2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr2/strided_slice/stackЖ
covtr2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr2/strided_slice/stack_1Ж
covtr2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr2/strided_slice/stack_2М
covtr2/strided_sliceStridedSlicecovtr2/Shape:output:0#covtr2/strided_slice/stack:output:0%covtr2/strided_slice/stack_1:output:0%covtr2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr2/strided_sliceb
covtr2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
covtr2/stack/1b
covtr2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
covtr2/stack/2b
covtr2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
covtr2/stack/3╝
covtr2/stackPackcovtr2/strided_slice:output:0covtr2/stack/1:output:0covtr2/stack/2:output:0covtr2/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr2/stackЖ
covtr2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr2/strided_slice_1/stackК
covtr2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr2/strided_slice_1/stack_1К
covtr2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr2/strided_slice_1/stack_2Ц
covtr2/strided_slice_1StridedSlicecovtr2/stack:output:0%covtr2/strided_slice_1/stack:output:0'covtr2/strided_slice_1/stack_1:output:0'covtr2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr2/strided_slice_1╚
&covtr2/conv2d_transpose/ReadVariableOpReadVariableOp/covtr2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02(
&covtr2/conv2d_transpose/ReadVariableOpМ
covtr2/conv2d_transposeConv2DBackpropInputcovtr2/stack:output:0.covtr2/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
covtr2/conv2d_transposeб
covtr2/BiasAdd/ReadVariableOpReadVariableOp&covtr2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
covtr2/BiasAdd/ReadVariableOpо
covtr2/BiasAddBiasAdd covtr2/conv2d_transpose:output:0%covtr2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
covtr2/BiasAddЕ
leaky_re_lu/LeakyRelu	LeakyRelucovtr2/BiasAdd:output:0*/
_output_shapes
:          2
leaky_re_lu/LeakyRelu░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp╢
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1▀
$batch_normalization/FusedBatchNormV3FusedBatchNormV3#leaky_re_lu/LeakyRelu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( 2&
$batch_normalization/FusedBatchNormV3Ф
dropout/IdentityIdentity(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
dropout/Identitye
covtr3/ShapeShapedropout/Identity:output:0*
T0*
_output_shapes
:2
covtr3/ShapeВ
covtr3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr3/strided_slice/stackЖ
covtr3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr3/strided_slice/stack_1Ж
covtr3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr3/strided_slice/stack_2М
covtr3/strided_sliceStridedSlicecovtr3/Shape:output:0#covtr3/strided_slice/stack:output:0%covtr3/strided_slice/stack_1:output:0%covtr3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr3/strided_sliceb
covtr3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :42
covtr3/stack/1b
covtr3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :,2
covtr3/stack/2b
covtr3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
covtr3/stack/3╝
covtr3/stackPackcovtr3/strided_slice:output:0covtr3/stack/1:output:0covtr3/stack/2:output:0covtr3/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr3/stackЖ
covtr3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr3/strided_slice_1/stackК
covtr3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr3/strided_slice_1/stack_1К
covtr3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr3/strided_slice_1/stack_2Ц
covtr3/strided_slice_1StridedSlicecovtr3/stack:output:0%covtr3/strided_slice_1/stack:output:0'covtr3/strided_slice_1/stack_1:output:0'covtr3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr3/strided_slice_1╚
&covtr3/conv2d_transpose/ReadVariableOpReadVariableOp/covtr3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02(
&covtr3/conv2d_transpose/ReadVariableOpН
covtr3/conv2d_transposeConv2DBackpropInputcovtr3/stack:output:0.covtr3/conv2d_transpose/ReadVariableOp:value:0dropout/Identity:output:0*
T0*/
_output_shapes
:         4,@*
paddingSAME*
strides
2
covtr3/conv2d_transposeб
covtr3/BiasAdd/ReadVariableOpReadVariableOp&covtr3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
covtr3/BiasAdd/ReadVariableOpо
covtr3/BiasAddBiasAdd covtr3/conv2d_transpose:output:0%covtr3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         4,@2
covtr3/BiasAddЙ
leaky_re_lu_1/LeakyRelu	LeakyRelucovtr3/BiasAdd:output:0*/
_output_shapes
:         4,@2
leaky_re_lu_1/LeakyRelu╢
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1э
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_1/LeakyRelu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         4,@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3Ъ
dropout_1/IdentityIdentity*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         4,@2
dropout_1/Identityg
covtr4/ShapeShapedropout_1/Identity:output:0*
T0*
_output_shapes
:2
covtr4/ShapeВ
covtr4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr4/strided_slice/stackЖ
covtr4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr4/strided_slice/stack_1Ж
covtr4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr4/strided_slice/stack_2М
covtr4/strided_sliceStridedSlicecovtr4/Shape:output:0#covtr4/strided_slice/stack:output:0%covtr4/strided_slice/stack_1:output:0%covtr4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr4/strided_sliceb
covtr4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :h2
covtr4/stack/1b
covtr4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :X2
covtr4/stack/2c
covtr4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :А2
covtr4/stack/3╝
covtr4/stackPackcovtr4/strided_slice:output:0covtr4/stack/1:output:0covtr4/stack/2:output:0covtr4/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr4/stackЖ
covtr4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr4/strided_slice_1/stackК
covtr4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr4/strided_slice_1/stack_1К
covtr4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr4/strided_slice_1/stack_2Ц
covtr4/strided_slice_1StridedSlicecovtr4/stack:output:0%covtr4/strided_slice_1/stack:output:0'covtr4/strided_slice_1/stack_1:output:0'covtr4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr4/strided_slice_1╔
&covtr4/conv2d_transpose/ReadVariableOpReadVariableOp/covtr4_conv2d_transpose_readvariableop_resource*'
_output_shapes
:А@*
dtype02(
&covtr4/conv2d_transpose/ReadVariableOpР
covtr4/conv2d_transposeConv2DBackpropInputcovtr4/stack:output:0.covtr4/conv2d_transpose/ReadVariableOp:value:0dropout_1/Identity:output:0*
T0*0
_output_shapes
:         hXА*
paddingSAME*
strides
2
covtr4/conv2d_transposeв
covtr4/BiasAdd/ReadVariableOpReadVariableOp&covtr4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
covtr4/BiasAdd/ReadVariableOpп
covtr4/BiasAddBiasAdd covtr4/conv2d_transpose:output:0%covtr4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         hXА2
covtr4/BiasAddК
leaky_re_lu_2/LeakyRelu	LeakyRelucovtr4/BiasAdd:output:0*0
_output_shapes
:         hXА2
leaky_re_lu_2/LeakyRelu╖
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$batch_normalization_2/ReadVariableOp╜
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype02(
&batch_normalization_2/ReadVariableOp_1ъ
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpЁ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Є
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_2/LeakyRelu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         hXА:А:А:А:А:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3Ы
dropout_2/IdentityIdentity*batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         hXА2
dropout_2/Identityc

cov3/ShapeShapedropout_2/Identity:output:0*
T0*
_output_shapes
:2

cov3/Shape~
cov3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cov3/strided_slice/stackВ
cov3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice/stack_1В
cov3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice/stack_2А
cov3/strided_sliceStridedSlicecov3/Shape:output:0!cov3/strided_slice/stack:output:0#cov3/strided_slice/stack_1:output:0#cov3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cov3/strided_slice^
cov3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :m2
cov3/stack/1^
cov3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Y2
cov3/stack/2^
cov3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
cov3/stack/3░

cov3/stackPackcov3/strided_slice:output:0cov3/stack/1:output:0cov3/stack/2:output:0cov3/stack/3:output:0*
N*
T0*
_output_shapes
:2

cov3/stackВ
cov3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cov3/strided_slice_1/stackЖ
cov3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice_1/stack_1Ж
cov3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice_1/stack_2К
cov3/strided_slice_1StridedSlicecov3/stack:output:0#cov3/strided_slice_1/stack:output:0%cov3/strided_slice_1/stack_1:output:0%cov3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cov3/strided_slice_1├
$cov3/conv2d_transpose/ReadVariableOpReadVariableOp-cov3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:А*
dtype02&
$cov3/conv2d_transpose/ReadVariableOpИ
cov3/conv2d_transposeConv2DBackpropInputcov3/stack:output:0,cov3/conv2d_transpose/ReadVariableOp:value:0dropout_2/Identity:output:0*
T0*/
_output_shapes
:         mY*
paddingVALID*
strides
2
cov3/conv2d_transposeЫ
cov3/BiasAdd/ReadVariableOpReadVariableOp$cov3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
cov3/BiasAdd/ReadVariableOpж
cov3/BiasAddBiasAddcov3/conv2d_transpose:output:0#cov3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         mY2
cov3/BiasAddx
cov3/SigmoidSigmoidcov3/BiasAdd:output:0*
T0*/
_output_shapes
:         mY2
cov3/Sigmoidl
IdentityIdentitycov3/Sigmoid:y:0*
T0*/
_output_shapes
:         mY2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:         d:::::::::::::::::::::::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Ш
Й
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1171716

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ж
Л
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1172012

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А:::::j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ы╙
Ї
F__inference_Generator_layer_call_and_return_conditional_losses_1172892

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource3
/covtr2_conv2d_transpose_readvariableop_resource*
&covtr2_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource3
/covtr3_conv2d_transpose_readvariableop_resource*
&covtr3_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource3
/covtr4_conv2d_transpose_readvariableop_resource*
&covtr4_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource1
-cov3_conv2d_transpose_readvariableop_resource(
$cov3_biasadd_readvariableop_resource
identityИв"batch_normalization/AssignNewValueв$batch_normalization/AssignNewValue_1в$batch_normalization_1/AssignNewValueв&batch_normalization_1/AssignNewValue_1в$batch_normalization_2/AssignNewValueв&batch_normalization_2/AssignNewValue_1а
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	dн*
dtype02
dense/MatMul/ReadVariableOpЖ
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         н2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:н*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         н2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:         н2

dense/Reluf
reshape/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
reshape/ShapeД
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackИ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1И
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2Т
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3ъ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeб
reshape/ReshapeReshapedense/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:         2
reshape/Reshaped
covtr2/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
covtr2/ShapeВ
covtr2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr2/strided_slice/stackЖ
covtr2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr2/strided_slice/stack_1Ж
covtr2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr2/strided_slice/stack_2М
covtr2/strided_sliceStridedSlicecovtr2/Shape:output:0#covtr2/strided_slice/stack:output:0%covtr2/strided_slice/stack_1:output:0%covtr2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr2/strided_sliceb
covtr2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
covtr2/stack/1b
covtr2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
covtr2/stack/2b
covtr2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
covtr2/stack/3╝
covtr2/stackPackcovtr2/strided_slice:output:0covtr2/stack/1:output:0covtr2/stack/2:output:0covtr2/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr2/stackЖ
covtr2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr2/strided_slice_1/stackК
covtr2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr2/strided_slice_1/stack_1К
covtr2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr2/strided_slice_1/stack_2Ц
covtr2/strided_slice_1StridedSlicecovtr2/stack:output:0%covtr2/strided_slice_1/stack:output:0'covtr2/strided_slice_1/stack_1:output:0'covtr2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr2/strided_slice_1╚
&covtr2/conv2d_transpose/ReadVariableOpReadVariableOp/covtr2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02(
&covtr2/conv2d_transpose/ReadVariableOpМ
covtr2/conv2d_transposeConv2DBackpropInputcovtr2/stack:output:0.covtr2/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
covtr2/conv2d_transposeб
covtr2/BiasAdd/ReadVariableOpReadVariableOp&covtr2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
covtr2/BiasAdd/ReadVariableOpо
covtr2/BiasAddBiasAdd covtr2/conv2d_transpose:output:0%covtr2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
covtr2/BiasAddЕ
leaky_re_lu/LeakyRelu	LeakyRelucovtr2/BiasAdd:output:0*/
_output_shapes
:          2
leaky_re_lu/LeakyRelu░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp╢
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1э
$batch_normalization/FusedBatchNormV3FusedBatchNormV3#leaky_re_lu/LeakyRelu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
exponential_avg_factor%═╠L>2&
$batch_normalization/FusedBatchNormV3ў
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValueЕ
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1s
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const╡
dropout/dropout/MulMul(batch_normalization/FusedBatchNormV3:y:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:          2
dropout/dropout/MulЖ
dropout/dropout/ShapeShape(batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout/dropout/Shape╘
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:          *
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yц
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:          2
dropout/dropout/GreaterEqualЯ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:          2
dropout/dropout/Castв
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:          2
dropout/dropout/Mul_1e
covtr3/ShapeShapedropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
covtr3/ShapeВ
covtr3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr3/strided_slice/stackЖ
covtr3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr3/strided_slice/stack_1Ж
covtr3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr3/strided_slice/stack_2М
covtr3/strided_sliceStridedSlicecovtr3/Shape:output:0#covtr3/strided_slice/stack:output:0%covtr3/strided_slice/stack_1:output:0%covtr3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr3/strided_sliceb
covtr3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :42
covtr3/stack/1b
covtr3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :,2
covtr3/stack/2b
covtr3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
covtr3/stack/3╝
covtr3/stackPackcovtr3/strided_slice:output:0covtr3/stack/1:output:0covtr3/stack/2:output:0covtr3/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr3/stackЖ
covtr3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr3/strided_slice_1/stackК
covtr3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr3/strided_slice_1/stack_1К
covtr3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr3/strided_slice_1/stack_2Ц
covtr3/strided_slice_1StridedSlicecovtr3/stack:output:0%covtr3/strided_slice_1/stack:output:0'covtr3/strided_slice_1/stack_1:output:0'covtr3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr3/strided_slice_1╚
&covtr3/conv2d_transpose/ReadVariableOpReadVariableOp/covtr3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02(
&covtr3/conv2d_transpose/ReadVariableOpН
covtr3/conv2d_transposeConv2DBackpropInputcovtr3/stack:output:0.covtr3/conv2d_transpose/ReadVariableOp:value:0dropout/dropout/Mul_1:z:0*
T0*/
_output_shapes
:         4,@*
paddingSAME*
strides
2
covtr3/conv2d_transposeб
covtr3/BiasAdd/ReadVariableOpReadVariableOp&covtr3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
covtr3/BiasAdd/ReadVariableOpо
covtr3/BiasAddBiasAdd covtr3/conv2d_transpose:output:0%covtr3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         4,@2
covtr3/BiasAddЙ
leaky_re_lu_1/LeakyRelu	LeakyRelucovtr3/BiasAdd:output:0*/
_output_shapes
:         4,@2
leaky_re_lu_1/LeakyRelu╢
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1√
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_1/LeakyRelu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         4,@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%═╠L>2(
&batch_normalization_1/FusedBatchNormV3Г
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValueС
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1w
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Const╜
dropout_1/dropout/MulMul*batch_normalization_1/FusedBatchNormV3:y:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:         4,@2
dropout_1/dropout/MulМ
dropout_1/dropout/ShapeShape*batch_normalization_1/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape┌
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:         4,@*
dtype020
.dropout_1/dropout/random_uniform/RandomUniformЙ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/yю
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         4,@2 
dropout_1/dropout/GreaterEqualе
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         4,@2
dropout_1/dropout/Castк
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:         4,@2
dropout_1/dropout/Mul_1g
covtr4/ShapeShapedropout_1/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
covtr4/ShapeВ
covtr4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr4/strided_slice/stackЖ
covtr4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr4/strided_slice/stack_1Ж
covtr4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr4/strided_slice/stack_2М
covtr4/strided_sliceStridedSlicecovtr4/Shape:output:0#covtr4/strided_slice/stack:output:0%covtr4/strided_slice/stack_1:output:0%covtr4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr4/strided_sliceb
covtr4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :h2
covtr4/stack/1b
covtr4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :X2
covtr4/stack/2c
covtr4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :А2
covtr4/stack/3╝
covtr4/stackPackcovtr4/strided_slice:output:0covtr4/stack/1:output:0covtr4/stack/2:output:0covtr4/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr4/stackЖ
covtr4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr4/strided_slice_1/stackК
covtr4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr4/strided_slice_1/stack_1К
covtr4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr4/strided_slice_1/stack_2Ц
covtr4/strided_slice_1StridedSlicecovtr4/stack:output:0%covtr4/strided_slice_1/stack:output:0'covtr4/strided_slice_1/stack_1:output:0'covtr4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr4/strided_slice_1╔
&covtr4/conv2d_transpose/ReadVariableOpReadVariableOp/covtr4_conv2d_transpose_readvariableop_resource*'
_output_shapes
:А@*
dtype02(
&covtr4/conv2d_transpose/ReadVariableOpР
covtr4/conv2d_transposeConv2DBackpropInputcovtr4/stack:output:0.covtr4/conv2d_transpose/ReadVariableOp:value:0dropout_1/dropout/Mul_1:z:0*
T0*0
_output_shapes
:         hXА*
paddingSAME*
strides
2
covtr4/conv2d_transposeв
covtr4/BiasAdd/ReadVariableOpReadVariableOp&covtr4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
covtr4/BiasAdd/ReadVariableOpп
covtr4/BiasAddBiasAdd covtr4/conv2d_transpose:output:0%covtr4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         hXА2
covtr4/BiasAddК
leaky_re_lu_2/LeakyRelu	LeakyRelucovtr4/BiasAdd:output:0*0
_output_shapes
:         hXА2
leaky_re_lu_2/LeakyRelu╖
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$batch_normalization_2/ReadVariableOp╜
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype02(
&batch_normalization_2/ReadVariableOp_1ъ
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpЁ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1А
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_2/LeakyRelu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         hXА:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%═╠L>2(
&batch_normalization_2/FusedBatchNormV3Г
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValueС
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1w
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/dropout/Const╛
dropout_2/dropout/MulMul*batch_normalization_2/FusedBatchNormV3:y:0 dropout_2/dropout/Const:output:0*
T0*0
_output_shapes
:         hXА2
dropout_2/dropout/MulМ
dropout_2/dropout/ShapeShape*batch_normalization_2/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape█
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*0
_output_shapes
:         hXА*
dtype020
.dropout_2/dropout/random_uniform/RandomUniformЙ
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_2/dropout/GreaterEqual/yя
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         hXА2 
dropout_2/dropout/GreaterEqualж
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         hXА2
dropout_2/dropout/Castл
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*0
_output_shapes
:         hXА2
dropout_2/dropout/Mul_1c

cov3/ShapeShapedropout_2/dropout/Mul_1:z:0*
T0*
_output_shapes
:2

cov3/Shape~
cov3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cov3/strided_slice/stackВ
cov3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice/stack_1В
cov3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice/stack_2А
cov3/strided_sliceStridedSlicecov3/Shape:output:0!cov3/strided_slice/stack:output:0#cov3/strided_slice/stack_1:output:0#cov3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cov3/strided_slice^
cov3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :m2
cov3/stack/1^
cov3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Y2
cov3/stack/2^
cov3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
cov3/stack/3░

cov3/stackPackcov3/strided_slice:output:0cov3/stack/1:output:0cov3/stack/2:output:0cov3/stack/3:output:0*
N*
T0*
_output_shapes
:2

cov3/stackВ
cov3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cov3/strided_slice_1/stackЖ
cov3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice_1/stack_1Ж
cov3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice_1/stack_2К
cov3/strided_slice_1StridedSlicecov3/stack:output:0#cov3/strided_slice_1/stack:output:0%cov3/strided_slice_1/stack_1:output:0%cov3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cov3/strided_slice_1├
$cov3/conv2d_transpose/ReadVariableOpReadVariableOp-cov3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:А*
dtype02&
$cov3/conv2d_transpose/ReadVariableOpИ
cov3/conv2d_transposeConv2DBackpropInputcov3/stack:output:0,cov3/conv2d_transpose/ReadVariableOp:value:0dropout_2/dropout/Mul_1:z:0*
T0*/
_output_shapes
:         mY*
paddingVALID*
strides
2
cov3/conv2d_transposeЫ
cov3/BiasAdd/ReadVariableOpReadVariableOp$cov3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
cov3/BiasAdd/ReadVariableOpж
cov3/BiasAddBiasAddcov3/conv2d_transpose:output:0#cov3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         mY2
cov3/BiasAddx
cov3/SigmoidSigmoidcov3/BiasAdd:output:0*
T0*/
_output_shapes
:         mY2
cov3/Sigmoid╪
IdentityIdentitycov3/Sigmoid:y:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1*
T0*/
_output_shapes
:         mY2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:         d::::::::::::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_1:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╟
}
(__inference_covtr2_layer_call_fn_1171623

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr2_layer_call_and_return_conditional_losses_11716132
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╠╕
├
"__inference__wrapped_model_1171579
	gen_noise2
.generator_dense_matmul_readvariableop_resource3
/generator_dense_biasadd_readvariableop_resource=
9generator_covtr2_conv2d_transpose_readvariableop_resource4
0generator_covtr2_biasadd_readvariableop_resource9
5generator_batch_normalization_readvariableop_resource;
7generator_batch_normalization_readvariableop_1_resourceJ
Fgenerator_batch_normalization_fusedbatchnormv3_readvariableop_resourceL
Hgenerator_batch_normalization_fusedbatchnormv3_readvariableop_1_resource=
9generator_covtr3_conv2d_transpose_readvariableop_resource4
0generator_covtr3_biasadd_readvariableop_resource;
7generator_batch_normalization_1_readvariableop_resource=
9generator_batch_normalization_1_readvariableop_1_resourceL
Hgenerator_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceN
Jgenerator_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource=
9generator_covtr4_conv2d_transpose_readvariableop_resource4
0generator_covtr4_biasadd_readvariableop_resource;
7generator_batch_normalization_2_readvariableop_resource=
9generator_batch_normalization_2_readvariableop_1_resourceL
Hgenerator_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceN
Jgenerator_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource;
7generator_cov3_conv2d_transpose_readvariableop_resource2
.generator_cov3_biasadd_readvariableop_resource
identityИ╛
%Generator/dense/MatMul/ReadVariableOpReadVariableOp.generator_dense_matmul_readvariableop_resource*
_output_shapes
:	dн*
dtype02'
%Generator/dense/MatMul/ReadVariableOpз
Generator/dense/MatMulMatMul	gen_noise-Generator/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         н2
Generator/dense/MatMul╜
&Generator/dense/BiasAdd/ReadVariableOpReadVariableOp/generator_dense_biasadd_readvariableop_resource*
_output_shapes	
:н*
dtype02(
&Generator/dense/BiasAdd/ReadVariableOp┬
Generator/dense/BiasAddBiasAdd Generator/dense/MatMul:product:0.Generator/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         н2
Generator/dense/BiasAddЙ
Generator/dense/ReluRelu Generator/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         н2
Generator/dense/ReluД
Generator/reshape/ShapeShape"Generator/dense/Relu:activations:0*
T0*
_output_shapes
:2
Generator/reshape/ShapeШ
%Generator/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Generator/reshape/strided_slice/stackЬ
'Generator/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/reshape/strided_slice/stack_1Ь
'Generator/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/reshape/strided_slice/stack_2╬
Generator/reshape/strided_sliceStridedSlice Generator/reshape/Shape:output:0.Generator/reshape/strided_slice/stack:output:00Generator/reshape/strided_slice/stack_1:output:00Generator/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
Generator/reshape/strided_sliceИ
!Generator/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!Generator/reshape/Reshape/shape/1И
!Generator/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!Generator/reshape/Reshape/shape/2И
!Generator/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!Generator/reshape/Reshape/shape/3ж
Generator/reshape/Reshape/shapePack(Generator/reshape/strided_slice:output:0*Generator/reshape/Reshape/shape/1:output:0*Generator/reshape/Reshape/shape/2:output:0*Generator/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
Generator/reshape/Reshape/shape╔
Generator/reshape/ReshapeReshape"Generator/dense/Relu:activations:0(Generator/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:         2
Generator/reshape/ReshapeВ
Generator/covtr2/ShapeShape"Generator/reshape/Reshape:output:0*
T0*
_output_shapes
:2
Generator/covtr2/ShapeЦ
$Generator/covtr2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr2/strided_slice/stackЪ
&Generator/covtr2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr2/strided_slice/stack_1Ъ
&Generator/covtr2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr2/strided_slice/stack_2╚
Generator/covtr2/strided_sliceStridedSliceGenerator/covtr2/Shape:output:0-Generator/covtr2/strided_slice/stack:output:0/Generator/covtr2/strided_slice/stack_1:output:0/Generator/covtr2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Generator/covtr2/strided_slicev
Generator/covtr2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr2/stack/1v
Generator/covtr2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr2/stack/2v
Generator/covtr2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Generator/covtr2/stack/3°
Generator/covtr2/stackPack'Generator/covtr2/strided_slice:output:0!Generator/covtr2/stack/1:output:0!Generator/covtr2/stack/2:output:0!Generator/covtr2/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr2/stackЪ
&Generator/covtr2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr2/strided_slice_1/stackЮ
(Generator/covtr2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr2/strided_slice_1/stack_1Ю
(Generator/covtr2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr2/strided_slice_1/stack_2╥
 Generator/covtr2/strided_slice_1StridedSliceGenerator/covtr2/stack:output:0/Generator/covtr2/strided_slice_1/stack:output:01Generator/covtr2/strided_slice_1/stack_1:output:01Generator/covtr2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr2/strided_slice_1ц
0Generator/covtr2/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype022
0Generator/covtr2/conv2d_transpose/ReadVariableOp╛
!Generator/covtr2/conv2d_transposeConv2DBackpropInputGenerator/covtr2/stack:output:08Generator/covtr2/conv2d_transpose/ReadVariableOp:value:0"Generator/reshape/Reshape:output:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2#
!Generator/covtr2/conv2d_transpose┐
'Generator/covtr2/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'Generator/covtr2/BiasAdd/ReadVariableOp╓
Generator/covtr2/BiasAddBiasAdd*Generator/covtr2/conv2d_transpose:output:0/Generator/covtr2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
Generator/covtr2/BiasAddг
Generator/leaky_re_lu/LeakyRelu	LeakyRelu!Generator/covtr2/BiasAdd:output:0*/
_output_shapes
:          2!
Generator/leaky_re_lu/LeakyRelu╬
,Generator/batch_normalization/ReadVariableOpReadVariableOp5generator_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02.
,Generator/batch_normalization/ReadVariableOp╘
.Generator/batch_normalization/ReadVariableOp_1ReadVariableOp7generator_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype020
.Generator/batch_normalization/ReadVariableOp_1Б
=Generator/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpFgenerator_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02?
=Generator/batch_normalization/FusedBatchNormV3/ReadVariableOpЗ
?Generator/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHgenerator_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02A
?Generator/batch_normalization/FusedBatchNormV3/ReadVariableOp_1е
.Generator/batch_normalization/FusedBatchNormV3FusedBatchNormV3-Generator/leaky_re_lu/LeakyRelu:activations:04Generator/batch_normalization/ReadVariableOp:value:06Generator/batch_normalization/ReadVariableOp_1:value:0EGenerator/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0GGenerator/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( 20
.Generator/batch_normalization/FusedBatchNormV3▓
Generator/dropout/IdentityIdentity2Generator/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2
Generator/dropout/IdentityГ
Generator/covtr3/ShapeShape#Generator/dropout/Identity:output:0*
T0*
_output_shapes
:2
Generator/covtr3/ShapeЦ
$Generator/covtr3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr3/strided_slice/stackЪ
&Generator/covtr3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr3/strided_slice/stack_1Ъ
&Generator/covtr3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr3/strided_slice/stack_2╚
Generator/covtr3/strided_sliceStridedSliceGenerator/covtr3/Shape:output:0-Generator/covtr3/strided_slice/stack:output:0/Generator/covtr3/strided_slice/stack_1:output:0/Generator/covtr3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Generator/covtr3/strided_slicev
Generator/covtr3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :42
Generator/covtr3/stack/1v
Generator/covtr3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :,2
Generator/covtr3/stack/2v
Generator/covtr3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Generator/covtr3/stack/3°
Generator/covtr3/stackPack'Generator/covtr3/strided_slice:output:0!Generator/covtr3/stack/1:output:0!Generator/covtr3/stack/2:output:0!Generator/covtr3/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr3/stackЪ
&Generator/covtr3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr3/strided_slice_1/stackЮ
(Generator/covtr3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr3/strided_slice_1/stack_1Ю
(Generator/covtr3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr3/strided_slice_1/stack_2╥
 Generator/covtr3/strided_slice_1StridedSliceGenerator/covtr3/stack:output:0/Generator/covtr3/strided_slice_1/stack:output:01Generator/covtr3/strided_slice_1/stack_1:output:01Generator/covtr3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr3/strided_slice_1ц
0Generator/covtr3/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype022
0Generator/covtr3/conv2d_transpose/ReadVariableOp┐
!Generator/covtr3/conv2d_transposeConv2DBackpropInputGenerator/covtr3/stack:output:08Generator/covtr3/conv2d_transpose/ReadVariableOp:value:0#Generator/dropout/Identity:output:0*
T0*/
_output_shapes
:         4,@*
paddingSAME*
strides
2#
!Generator/covtr3/conv2d_transpose┐
'Generator/covtr3/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'Generator/covtr3/BiasAdd/ReadVariableOp╓
Generator/covtr3/BiasAddBiasAdd*Generator/covtr3/conv2d_transpose:output:0/Generator/covtr3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         4,@2
Generator/covtr3/BiasAddз
!Generator/leaky_re_lu_1/LeakyRelu	LeakyRelu!Generator/covtr3/BiasAdd:output:0*/
_output_shapes
:         4,@2#
!Generator/leaky_re_lu_1/LeakyRelu╘
.Generator/batch_normalization_1/ReadVariableOpReadVariableOp7generator_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype020
.Generator/batch_normalization_1/ReadVariableOp┌
0Generator/batch_normalization_1/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0Generator/batch_normalization_1/ReadVariableOp_1З
?Generator/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02A
?Generator/batch_normalization_1/FusedBatchNormV3/ReadVariableOpН
AGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02C
AGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1│
0Generator/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_1/LeakyRelu:activations:06Generator/batch_normalization_1/ReadVariableOp:value:08Generator/batch_normalization_1/ReadVariableOp_1:value:0GGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         4,@:@:@:@:@:*
epsilon%oГ:*
is_training( 22
0Generator/batch_normalization_1/FusedBatchNormV3╕
Generator/dropout_1/IdentityIdentity4Generator/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         4,@2
Generator/dropout_1/IdentityЕ
Generator/covtr4/ShapeShape%Generator/dropout_1/Identity:output:0*
T0*
_output_shapes
:2
Generator/covtr4/ShapeЦ
$Generator/covtr4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr4/strided_slice/stackЪ
&Generator/covtr4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr4/strided_slice/stack_1Ъ
&Generator/covtr4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr4/strided_slice/stack_2╚
Generator/covtr4/strided_sliceStridedSliceGenerator/covtr4/Shape:output:0-Generator/covtr4/strided_slice/stack:output:0/Generator/covtr4/strided_slice/stack_1:output:0/Generator/covtr4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Generator/covtr4/strided_slicev
Generator/covtr4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :h2
Generator/covtr4/stack/1v
Generator/covtr4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :X2
Generator/covtr4/stack/2w
Generator/covtr4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :А2
Generator/covtr4/stack/3°
Generator/covtr4/stackPack'Generator/covtr4/strided_slice:output:0!Generator/covtr4/stack/1:output:0!Generator/covtr4/stack/2:output:0!Generator/covtr4/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr4/stackЪ
&Generator/covtr4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr4/strided_slice_1/stackЮ
(Generator/covtr4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr4/strided_slice_1/stack_1Ю
(Generator/covtr4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr4/strided_slice_1/stack_2╥
 Generator/covtr4/strided_slice_1StridedSliceGenerator/covtr4/stack:output:0/Generator/covtr4/strided_slice_1/stack:output:01Generator/covtr4/strided_slice_1/stack_1:output:01Generator/covtr4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr4/strided_slice_1ч
0Generator/covtr4/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr4_conv2d_transpose_readvariableop_resource*'
_output_shapes
:А@*
dtype022
0Generator/covtr4/conv2d_transpose/ReadVariableOp┬
!Generator/covtr4/conv2d_transposeConv2DBackpropInputGenerator/covtr4/stack:output:08Generator/covtr4/conv2d_transpose/ReadVariableOp:value:0%Generator/dropout_1/Identity:output:0*
T0*0
_output_shapes
:         hXА*
paddingSAME*
strides
2#
!Generator/covtr4/conv2d_transpose└
'Generator/covtr4/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02)
'Generator/covtr4/BiasAdd/ReadVariableOp╫
Generator/covtr4/BiasAddBiasAdd*Generator/covtr4/conv2d_transpose:output:0/Generator/covtr4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         hXА2
Generator/covtr4/BiasAddи
!Generator/leaky_re_lu_2/LeakyRelu	LeakyRelu!Generator/covtr4/BiasAdd:output:0*0
_output_shapes
:         hXА2#
!Generator/leaky_re_lu_2/LeakyRelu╒
.Generator/batch_normalization_2/ReadVariableOpReadVariableOp7generator_batch_normalization_2_readvariableop_resource*
_output_shapes	
:А*
dtype020
.Generator/batch_normalization_2/ReadVariableOp█
0Generator/batch_normalization_2/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype022
0Generator/batch_normalization_2/ReadVariableOp_1И
?Generator/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02A
?Generator/batch_normalization_2/FusedBatchNormV3/ReadVariableOpО
AGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02C
AGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1╕
0Generator/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_2/LeakyRelu:activations:06Generator/batch_normalization_2/ReadVariableOp:value:08Generator/batch_normalization_2/ReadVariableOp_1:value:0GGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         hXА:А:А:А:А:*
epsilon%oГ:*
is_training( 22
0Generator/batch_normalization_2/FusedBatchNormV3╣
Generator/dropout_2/IdentityIdentity4Generator/batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         hXА2
Generator/dropout_2/IdentityБ
Generator/cov3/ShapeShape%Generator/dropout_2/Identity:output:0*
T0*
_output_shapes
:2
Generator/cov3/ShapeТ
"Generator/cov3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"Generator/cov3/strided_slice/stackЦ
$Generator/cov3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$Generator/cov3/strided_slice/stack_1Ц
$Generator/cov3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$Generator/cov3/strided_slice/stack_2╝
Generator/cov3/strided_sliceStridedSliceGenerator/cov3/Shape:output:0+Generator/cov3/strided_slice/stack:output:0-Generator/cov3/strided_slice/stack_1:output:0-Generator/cov3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Generator/cov3/strided_slicer
Generator/cov3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :m2
Generator/cov3/stack/1r
Generator/cov3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Y2
Generator/cov3/stack/2r
Generator/cov3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/cov3/stack/3ь
Generator/cov3/stackPack%Generator/cov3/strided_slice:output:0Generator/cov3/stack/1:output:0Generator/cov3/stack/2:output:0Generator/cov3/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/cov3/stackЦ
$Generator/cov3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/cov3/strided_slice_1/stackЪ
&Generator/cov3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/cov3/strided_slice_1/stack_1Ъ
&Generator/cov3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/cov3/strided_slice_1/stack_2╞
Generator/cov3/strided_slice_1StridedSliceGenerator/cov3/stack:output:0-Generator/cov3/strided_slice_1/stack:output:0/Generator/cov3/strided_slice_1/stack_1:output:0/Generator/cov3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Generator/cov3/strided_slice_1с
.Generator/cov3/conv2d_transpose/ReadVariableOpReadVariableOp7generator_cov3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:А*
dtype020
.Generator/cov3/conv2d_transpose/ReadVariableOp║
Generator/cov3/conv2d_transposeConv2DBackpropInputGenerator/cov3/stack:output:06Generator/cov3/conv2d_transpose/ReadVariableOp:value:0%Generator/dropout_2/Identity:output:0*
T0*/
_output_shapes
:         mY*
paddingVALID*
strides
2!
Generator/cov3/conv2d_transpose╣
%Generator/cov3/BiasAdd/ReadVariableOpReadVariableOp.generator_cov3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Generator/cov3/BiasAdd/ReadVariableOp╬
Generator/cov3/BiasAddBiasAdd(Generator/cov3/conv2d_transpose:output:0-Generator/cov3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         mY2
Generator/cov3/BiasAddЦ
Generator/cov3/SigmoidSigmoidGenerator/cov3/BiasAdd:output:0*
T0*/
_output_shapes
:         mY2
Generator/cov3/Sigmoidv
IdentityIdentityGenerator/cov3/Sigmoid:y:0*
T0*/
_output_shapes
:         mY2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:         d:::::::::::::::::::::::R N
'
_output_shapes
:         d
#
_user_specified_name	gen_noise
╢
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_1172361

inputs

identity_1u
IdentityIdentityinputs*
T0*B
_output_shapes0
.:,                           А2

IdentityД

Identity_1IdentityIdentity:output:0*
T0*B
_output_shapes0
.:,                           А2

Identity_1"!

identity_1Identity_1:output:0*A
_input_shapes0
.:,                           А:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ь7
Ю

 __inference__traced_save_1173567
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop,
(savev2_covtr2_kernel_read_readvariableop*
&savev2_covtr2_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop,
(savev2_covtr3_kernel_read_readvariableop*
&savev2_covtr3_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop,
(savev2_covtr4_kernel_read_readvariableop*
&savev2_covtr4_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop*
&savev2_cov3_kernel_read_readvariableop(
$savev2_cov3_bias_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_8ff101327a6a4ea2b0cdaf73f05f7746/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameИ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ъ

valueР
BН
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names╢
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesй

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop(savev2_covtr2_kernel_read_readvariableop&savev2_covtr2_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop(savev2_covtr3_kernel_read_readvariableop&savev2_covtr3_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop(savev2_covtr4_kernel_read_readvariableop&savev2_covtr4_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop&savev2_cov3_kernel_read_readvariableop$savev2_cov3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*┌
_input_shapes╚
┼: :	dн:н: : : : : : :@ :@:@:@:@:@:А@:А:А:А:А:А:А:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	dн:!

_output_shapes	
:н:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,	(
&
_output_shapes
:@ : 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:А@:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:-)
'
_output_shapes
:А: 

_output_shapes
::

_output_shapes
: 
▐
e
F__inference_dropout_2_layer_call_and_return_conditional_losses_1173463

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/ConstО
dropout/MulMulinputsdropout/Const:output:0*
T0*B
_output_shapes0
.:,                           А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╧
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*B
_output_shapes0
.:,                           А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┘
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*B
_output_shapes0
.:,                           А2
dropout/GreaterEqualЪ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*B
_output_shapes0
.:,                           А2
dropout/CastХ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*B
_output_shapes0
.:,                           А2
dropout/Mul_1А
IdentityIdentitydropout/Mul_1:z:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           А:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╩
п
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1173306

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%═╠L>2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Ї!
╡
C__inference_covtr2_layer_call_and_return_conditional_losses_1171613

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ь
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3В
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ь
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOpЁ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpд
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           :::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
н
к
B__inference_dense_layer_call_and_return_conditional_losses_1173147

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	dн*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         н2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:н*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         н2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         н2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         н2

Identity"
identityIdentity:output:0*.
_input_shapes
:         d:::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
░
b
D__inference_dropout_layer_call_and_return_conditional_losses_1172195

inputs

identity_1t
IdentityIdentityinputs*
T0*A
_output_shapes/
-:+                            2

IdentityГ

Identity_1IdentityIdentity:output:0*
T0*A
_output_shapes/
-:+                            2

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ї!
╡
C__inference_covtr3_layer_call_and_return_conditional_losses_1171761

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ь
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3В
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ь
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2d_transpose/ReadVariableOpЁ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpд
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            :::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
М
K
/__inference_leaky_re_lu_1_layer_call_fn_1173286

inputs
identityч
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_11722182
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╒
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_1172273

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/ConstН
dropout/MulMulinputsdropout/Const:output:0*
T0*A
_output_shapes/
-:+                           @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*A
_output_shapes/
-:+                           @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y╪
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*A
_output_shapes/
-:+                           @2
dropout/GreaterEqualЩ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+                           @2
dropout/CastФ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*A
_output_shapes/
-:+                           @2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╣
║
+__inference_Generator_layer_call_fn_1172668
	gen_noise
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityИвStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCall	gen_noiseunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_Generator_layer_call_and_return_conditional_losses_11726212
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:         d::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:         d
#
_user_specified_name	gen_noise
ЄL
▌
F__inference_Generator_layer_call_and_return_conditional_losses_1172509

inputs
dense_1172449
dense_1172451
covtr2_1172455
covtr2_1172457
batch_normalization_1172461
batch_normalization_1172463
batch_normalization_1172465
batch_normalization_1172467
covtr3_1172471
covtr3_1172473!
batch_normalization_1_1172477!
batch_normalization_1_1172479!
batch_normalization_1_1172481!
batch_normalization_1_1172483
covtr4_1172487
covtr4_1172489!
batch_normalization_2_1172493!
batch_normalization_2_1172495!
batch_normalization_2_1172497!
batch_normalization_2_1172499
cov3_1172503
cov3_1172505
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallвcov3/StatefulPartitionedCallвcovtr2/StatefulPartitionedCallвcovtr3/StatefulPartitionedCallвcovtr4/StatefulPartitionedCallвdense/StatefulPartitionedCallвdropout/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallО
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1172449dense_1172451*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         н*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_11720872
dense/StatefulPartitionedCall 
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_11721172
reshape/PartitionedCall╞
covtr2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr2_1172455covtr2_1172457*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr2_layer_call_and_return_conditional_losses_11716132 
covtr2/StatefulPartitionedCallЮ
leaky_re_lu/PartitionedCallPartitionedCall'covtr2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_11721352
leaky_re_lu/PartitionedCall╟
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_1172461batch_normalization_1172463batch_normalization_1172465batch_normalization_1172467*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_11716852-
+batch_normalization/StatefulPartitionedCall╖
dropout/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_11721902!
dropout/StatefulPartitionedCall╬
covtr3/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0covtr3_1172471covtr3_1172473*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr3_layer_call_and_return_conditional_losses_11717612 
covtr3/StatefulPartitionedCallд
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_11722182
leaky_re_lu_1/PartitionedCall╫
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_1172477batch_normalization_1_1172479batch_normalization_1_1172481batch_normalization_1_1172483*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11718332/
-batch_normalization_1/StatefulPartitionedCallс
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_11722732#
!dropout_1/StatefulPartitionedCall╤
covtr4/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0covtr4_1172487covtr4_1172489*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr4_layer_call_and_return_conditional_losses_11719092 
covtr4/StatefulPartitionedCallе
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_11723012
leaky_re_lu_2/PartitionedCall╪
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_1172493batch_normalization_2_1172495batch_normalization_2_1172497batch_normalization_2_1172499*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11719812/
-batch_normalization_2/StatefulPartitionedCallф
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_11723562#
!dropout_2/StatefulPartitionedCall╞
cov3/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0cov3_1172503cov3_1172505*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_cov3_layer_call_and_return_conditional_losses_11720622
cov3/StatefulPartitionedCallн
IdentityIdentity%cov3/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^cov3/StatefulPartitionedCall^covtr2/StatefulPartitionedCall^covtr3/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:         d::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2<
cov3/StatefulPartitionedCallcov3/StatefulPartitionedCall2@
covtr2/StatefulPartitionedCallcovtr2/StatefulPartitionedCall2@
covtr3/StatefulPartitionedCallcovtr3/StatefulPartitionedCall2@
covtr4/StatefulPartitionedCallcovtr4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
√L
р
F__inference_Generator_layer_call_and_return_conditional_losses_1172380
	gen_noise
dense_1172098
dense_1172100
covtr2_1172125
covtr2_1172127
batch_normalization_1172169
batch_normalization_1172171
batch_normalization_1172173
batch_normalization_1172175
covtr3_1172208
covtr3_1172210!
batch_normalization_1_1172252!
batch_normalization_1_1172254!
batch_normalization_1_1172256!
batch_normalization_1_1172258
covtr4_1172291
covtr4_1172293!
batch_normalization_2_1172335!
batch_normalization_2_1172337!
batch_normalization_2_1172339!
batch_normalization_2_1172341
cov3_1172374
cov3_1172376
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallвcov3/StatefulPartitionedCallвcovtr2/StatefulPartitionedCallвcovtr3/StatefulPartitionedCallвcovtr4/StatefulPartitionedCallвdense/StatefulPartitionedCallвdropout/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallС
dense/StatefulPartitionedCallStatefulPartitionedCall	gen_noisedense_1172098dense_1172100*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         н*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_11720872
dense/StatefulPartitionedCall 
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_11721172
reshape/PartitionedCall╞
covtr2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr2_1172125covtr2_1172127*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr2_layer_call_and_return_conditional_losses_11716132 
covtr2/StatefulPartitionedCallЮ
leaky_re_lu/PartitionedCallPartitionedCall'covtr2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_11721352
leaky_re_lu/PartitionedCall╟
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_1172169batch_normalization_1172171batch_normalization_1172173batch_normalization_1172175*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_11716852-
+batch_normalization/StatefulPartitionedCall╖
dropout/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_11721902!
dropout/StatefulPartitionedCall╬
covtr3/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0covtr3_1172208covtr3_1172210*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr3_layer_call_and_return_conditional_losses_11717612 
covtr3/StatefulPartitionedCallд
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_11722182
leaky_re_lu_1/PartitionedCall╫
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_1172252batch_normalization_1_1172254batch_normalization_1_1172256batch_normalization_1_1172258*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11718332/
-batch_normalization_1/StatefulPartitionedCallс
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_11722732#
!dropout_1/StatefulPartitionedCall╤
covtr4/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0covtr4_1172291covtr4_1172293*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr4_layer_call_and_return_conditional_losses_11719092 
covtr4/StatefulPartitionedCallе
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_11723012
leaky_re_lu_2/PartitionedCall╪
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_1172335batch_normalization_2_1172337batch_normalization_2_1172339batch_normalization_2_1172341*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11719812/
-batch_normalization_2/StatefulPartitionedCallф
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_11723562#
!dropout_2/StatefulPartitionedCall╞
cov3/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0cov3_1172374cov3_1172376*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_cov3_layer_call_and_return_conditional_losses_11720622
cov3/StatefulPartitionedCallн
IdentityIdentity%cov3/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^cov3/StatefulPartitionedCall^covtr2/StatefulPartitionedCall^covtr3/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:         d::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2<
cov3/StatefulPartitionedCallcov3/StatefulPartitionedCall2@
covtr2/StatefulPartitionedCallcovtr2/StatefulPartitionedCall2@
covtr3/StatefulPartitionedCallcovtr3/StatefulPartitionedCall2@
covtr4/StatefulPartitionedCallcovtr4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:R N
'
_output_shapes
:         d
#
_user_specified_name	gen_noise
╝%
│
A__inference_cov3_layer_call_and_return_conditional_losses_1172062

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ь
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3В
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ь
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3┤
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:А*
dtype02!
conv2d_transpose/ReadVariableOpё
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpд
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2	
Sigmoidy
IdentityIdentitySigmoid:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,                           А:::j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╙
c
D__inference_dropout_layer_call_and_return_conditional_losses_1173261

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/ConstН
dropout/MulMulinputsdropout/Const:output:0*
T0*A
_output_shapes/
-:+                            2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*A
_output_shapes/
-:+                            *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y╪
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*A
_output_shapes/
-:+                            2
dropout/GreaterEqualЩ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+                            2
dropout/CastФ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*A
_output_shapes/
-:+                            2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ы
┤
%__inference_signature_wrapper_1172719
	gen_noise
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityИвStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCall	gen_noiseunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         mY*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *+
f&R$
"__inference__wrapped_model_11715792
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         mY2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:         d::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:         d
#
_user_specified_name	gen_noise
кH
є
F__inference_Generator_layer_call_and_return_conditional_losses_1172621

inputs
dense_1172561
dense_1172563
covtr2_1172567
covtr2_1172569
batch_normalization_1172573
batch_normalization_1172575
batch_normalization_1172577
batch_normalization_1172579
covtr3_1172583
covtr3_1172585!
batch_normalization_1_1172589!
batch_normalization_1_1172591!
batch_normalization_1_1172593!
batch_normalization_1_1172595
covtr4_1172599
covtr4_1172601!
batch_normalization_2_1172605!
batch_normalization_2_1172607!
batch_normalization_2_1172609!
batch_normalization_2_1172611
cov3_1172615
cov3_1172617
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallвcov3/StatefulPartitionedCallвcovtr2/StatefulPartitionedCallвcovtr3/StatefulPartitionedCallвcovtr4/StatefulPartitionedCallвdense/StatefulPartitionedCallО
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1172561dense_1172563*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         н*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_11720872
dense/StatefulPartitionedCall 
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_11721172
reshape/PartitionedCall╞
covtr2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr2_1172567covtr2_1172569*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr2_layer_call_and_return_conditional_losses_11716132 
covtr2/StatefulPartitionedCallЮ
leaky_re_lu/PartitionedCallPartitionedCall'covtr2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_11721352
leaky_re_lu/PartitionedCall╔
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_1172573batch_normalization_1172575batch_normalization_1172577batch_normalization_1172579*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_11717162-
+batch_normalization/StatefulPartitionedCallЯ
dropout/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_11721952
dropout/PartitionedCall╞
covtr3/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0covtr3_1172583covtr3_1172585*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr3_layer_call_and_return_conditional_losses_11717612 
covtr3/StatefulPartitionedCallд
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_11722182
leaky_re_lu_1/PartitionedCall┘
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_1172589batch_normalization_1_1172591batch_normalization_1_1172593batch_normalization_1_1172595*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11718642/
-batch_normalization_1/StatefulPartitionedCallз
dropout_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_11722782
dropout_1/PartitionedCall╔
covtr4/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0covtr4_1172599covtr4_1172601*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr4_layer_call_and_return_conditional_losses_11719092 
covtr4/StatefulPartitionedCallе
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_11723012
leaky_re_lu_2/PartitionedCall┌
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_1172605batch_normalization_2_1172607batch_normalization_2_1172609batch_normalization_2_1172611*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11720122/
-batch_normalization_2/StatefulPartitionedCallи
dropout_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_11723612
dropout_2/PartitionedCall╛
cov3/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0cov3_1172615cov3_1172617*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_cov3_layer_call_and_return_conditional_losses_11720622
cov3/StatefulPartitionedCall├
IdentityIdentity%cov3/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^cov3/StatefulPartitionedCall^covtr2/StatefulPartitionedCall^covtr3/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:         d::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2<
cov3/StatefulPartitionedCallcov3/StatefulPartitionedCall2@
covtr2/StatefulPartitionedCallcovtr2/StatefulPartitionedCall2@
covtr3/StatefulPartitionedCallcovtr3/StatefulPartitionedCall2@
covtr4/StatefulPartitionedCallcovtr4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
░
f
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_1172301

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,                           А2
	LeakyReluЖ
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           А:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ъ
Л
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1173324

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @:::::i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
·!
╡
C__inference_covtr4_layer_call_and_return_conditional_losses_1171909

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ь
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :А2	
stack/3В
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ь
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3┤
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:А@*
dtype02!
conv2d_transpose/ReadVariableOpё
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
conv2d_transposeН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           @:::i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
▀
|
'__inference_dense_layer_call_fn_1173156

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         н*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_11720872
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         н2

Identity"
identityIdentity:output:0*.
_input_shapes
:         d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Д
G
+__inference_dropout_1_layer_call_fn_1173377

inputs
identityу
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_11722782
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ч
`
D__inference_reshape_layer_call_and_return_conditional_losses_1172117

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3║
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*'
_input_shapes
:         н:P L
(
_output_shapes
:         н
 
_user_specified_nameinputs
к
к
7__inference_batch_normalization_2_layer_call_fn_1173451

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall╝
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11720122
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
▓
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_1172278

inputs

identity_1t
IdentityIdentityinputs*
T0*A
_output_shapes/
-:+                           @2

IdentityГ

Identity_1IdentityIdentity:output:0*
T0*A
_output_shapes/
-:+                           @2

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+                           @:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╚
н
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1173205

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%═╠L>2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╙
c
D__inference_dropout_layer_call_and_return_conditional_losses_1172190

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/ConstН
dropout/MulMulinputsdropout/Const:output:0*
T0*A
_output_shapes/
-:+                            2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*A
_output_shapes/
-:+                            *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y╪
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*A
_output_shapes/
-:+                            2
dropout/GreaterEqualЩ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+                            2
dropout/CastФ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*A
_output_shapes/
-:+                            2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ч
`
D__inference_reshape_layer_call_and_return_conditional_losses_1173170

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3║
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*'
_input_shapes
:         н:P L
(
_output_shapes
:         н
 
_user_specified_nameinputs
а
и
5__inference_batch_normalization_layer_call_fn_1173236

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall╖
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_11716852
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
│H
Ў
F__inference_Generator_layer_call_and_return_conditional_losses_1172443
	gen_noise
dense_1172383
dense_1172385
covtr2_1172389
covtr2_1172391
batch_normalization_1172395
batch_normalization_1172397
batch_normalization_1172399
batch_normalization_1172401
covtr3_1172405
covtr3_1172407!
batch_normalization_1_1172411!
batch_normalization_1_1172413!
batch_normalization_1_1172415!
batch_normalization_1_1172417
covtr4_1172421
covtr4_1172423!
batch_normalization_2_1172427!
batch_normalization_2_1172429!
batch_normalization_2_1172431!
batch_normalization_2_1172433
cov3_1172437
cov3_1172439
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallвcov3/StatefulPartitionedCallвcovtr2/StatefulPartitionedCallвcovtr3/StatefulPartitionedCallвcovtr4/StatefulPartitionedCallвdense/StatefulPartitionedCallС
dense/StatefulPartitionedCallStatefulPartitionedCall	gen_noisedense_1172383dense_1172385*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         н*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_11720872
dense/StatefulPartitionedCall 
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_11721172
reshape/PartitionedCall╞
covtr2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr2_1172389covtr2_1172391*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr2_layer_call_and_return_conditional_losses_11716132 
covtr2/StatefulPartitionedCallЮ
leaky_re_lu/PartitionedCallPartitionedCall'covtr2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_11721352
leaky_re_lu/PartitionedCall╔
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_1172395batch_normalization_1172397batch_normalization_1172399batch_normalization_1172401*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_11717162-
+batch_normalization/StatefulPartitionedCallЯ
dropout/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_11721952
dropout/PartitionedCall╞
covtr3/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0covtr3_1172405covtr3_1172407*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr3_layer_call_and_return_conditional_losses_11717612 
covtr3/StatefulPartitionedCallд
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_11722182
leaky_re_lu_1/PartitionedCall┘
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_1172411batch_normalization_1_1172413batch_normalization_1_1172415batch_normalization_1_1172417*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11718642/
-batch_normalization_1/StatefulPartitionedCallз
dropout_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_11722782
dropout_1/PartitionedCall╔
covtr4/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0covtr4_1172421covtr4_1172423*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr4_layer_call_and_return_conditional_losses_11719092 
covtr4/StatefulPartitionedCallе
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_11723012
leaky_re_lu_2/PartitionedCall┌
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_1172427batch_normalization_2_1172429batch_normalization_2_1172431batch_normalization_2_1172433*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11720122/
-batch_normalization_2/StatefulPartitionedCallи
dropout_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_11723612
dropout_2/PartitionedCall╛
cov3/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0cov3_1172437cov3_1172439*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_cov3_layer_call_and_return_conditional_losses_11720622
cov3/StatefulPartitionedCall├
IdentityIdentity%cov3/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^cov3/StatefulPartitionedCall^covtr2/StatefulPartitionedCall^covtr3/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:         d::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2<
cov3/StatefulPartitionedCallcov3/StatefulPartitionedCall2@
covtr2/StatefulPartitionedCallcovtr2/StatefulPartitionedCall2@
covtr3/StatefulPartitionedCallcovtr3/StatefulPartitionedCall2@
covtr4/StatefulPartitionedCallcovtr4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:R N
'
_output_shapes
:         d
#
_user_specified_name	gen_noise
й
E
)__inference_reshape_layer_call_fn_1173175

inputs
identity╧
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_11721172
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*'
_input_shapes
:         н:P L
(
_output_shapes
:         н
 
_user_specified_nameinputs
к
d
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_1172135

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                            2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
в
и
5__inference_batch_normalization_layer_call_fn_1173249

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_11717162
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
░
f
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_1173382

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,                           А2
	LeakyReluЖ
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           А:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
░
b
D__inference_dropout_layer_call_and_return_conditional_losses_1173266

inputs

identity_1t
IdentityIdentityinputs*
T0*A
_output_shapes/
-:+                            2

IdentityГ

Identity_1IdentityIdentity:output:0*
T0*A
_output_shapes/
-:+                            2

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
к
d
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_1173180

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                            2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╒
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_1173362

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/ConstН
dropout/MulMulinputsdropout/Const:output:0*
T0*A
_output_shapes/
-:+                           @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*A
_output_shapes/
-:+                           @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y╪
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*A
_output_shapes/
-:+                           @2
dropout/GreaterEqualЩ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+                           @2
dropout/CastФ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*A
_output_shapes/
-:+                           @2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
П
d
+__inference_dropout_1_layer_call_fn_1173372

inputs
identityИвStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_11722732
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╔
}
(__inference_covtr4_layer_call_fn_1171919

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr4_layer_call_and_return_conditional_losses_11719092
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
У
d
+__inference_dropout_2_layer_call_fn_1173473

inputs
identityИвStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_11723562
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           А22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
│
║
+__inference_Generator_layer_call_fn_1172556
	gen_noise
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityИвStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCall	gen_noiseunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_Generator_layer_call_and_return_conditional_losses_11725092
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:         d::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:         d
#
_user_specified_name	gen_noise
╟
}
(__inference_covtr3_layer_call_fn_1171771

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr3_layer_call_and_return_conditional_losses_11717612
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
и
к
7__inference_batch_normalization_2_layer_call_fn_1173438

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11719812
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╢
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_1173468

inputs

identity_1u
IdentityIdentityinputs*
T0*B
_output_shapes0
.:,                           А2

IdentityД

Identity_1IdentityIdentity:output:0*
T0*B
_output_shapes0
.:,                           А2

Identity_1"!

identity_1Identity_1:output:0*A
_input_shapes0
.:,                           А:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╡_
н
#__inference__traced_restore_1173643
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias$
 assignvariableop_2_covtr2_kernel"
assignvariableop_3_covtr2_bias0
,assignvariableop_4_batch_normalization_gamma/
+assignvariableop_5_batch_normalization_beta6
2assignvariableop_6_batch_normalization_moving_mean:
6assignvariableop_7_batch_normalization_moving_variance$
 assignvariableop_8_covtr3_kernel"
assignvariableop_9_covtr3_bias3
/assignvariableop_10_batch_normalization_1_gamma2
.assignvariableop_11_batch_normalization_1_beta9
5assignvariableop_12_batch_normalization_1_moving_mean=
9assignvariableop_13_batch_normalization_1_moving_variance%
!assignvariableop_14_covtr4_kernel#
assignvariableop_15_covtr4_bias3
/assignvariableop_16_batch_normalization_2_gamma2
.assignvariableop_17_batch_normalization_2_beta9
5assignvariableop_18_batch_normalization_2_moving_mean=
9assignvariableop_19_batch_normalization_2_moving_variance#
assignvariableop_20_cov3_kernel!
assignvariableop_21_cov3_bias
identity_23ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9О
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ъ

valueР
BН
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names╝
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЮ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЬ
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1в
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2е
AssignVariableOp_2AssignVariableOp assignvariableop_2_covtr2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3г
AssignVariableOp_3AssignVariableOpassignvariableop_3_covtr2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4▒
AssignVariableOp_4AssignVariableOp,assignvariableop_4_batch_normalization_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5░
AssignVariableOp_5AssignVariableOp+assignvariableop_5_batch_normalization_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6╖
AssignVariableOp_6AssignVariableOp2assignvariableop_6_batch_normalization_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7╗
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8е
AssignVariableOp_8AssignVariableOp assignvariableop_8_covtr3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9г
AssignVariableOp_9AssignVariableOpassignvariableop_9_covtr3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10╖
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_1_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11╢
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_1_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12╜
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_1_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13┴
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_1_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14й
AssignVariableOp_14AssignVariableOp!assignvariableop_14_covtr4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15з
AssignVariableOp_15AssignVariableOpassignvariableop_15_covtr4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16╖
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_2_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17╢
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_2_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18╜
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_2_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19┴
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_2_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20з
AssignVariableOp_20AssignVariableOpassignvariableop_20_cov3_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21е
AssignVariableOp_21AssignVariableOpassignvariableop_21_cov3_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_219
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp┬
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22╡
Identity_23IdentityIdentity_22:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_23"#
identity_23Identity_23:output:0*m
_input_shapes\
Z: ::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╚
н
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1171685

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%═╠L>2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
н
к
B__inference_dense_layer_call_and_return_conditional_losses_1172087

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	dн*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         н2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:н*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         н2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         н2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         н2

Identity"
identityIdentity:output:0*.
_input_shapes
:         d:::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
И
I
-__inference_leaky_re_lu_layer_call_fn_1173185

inputs
identityх
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_11721352
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Р
K
/__inference_leaky_re_lu_2_layer_call_fn_1173387

inputs
identityш
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_11723012
PartitionedCallЗ
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           А:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
▓
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_1173367

inputs

identity_1t
IdentityIdentityinputs*
T0*A
_output_shapes/
-:+                           @2

IdentityГ

Identity_1IdentityIdentity:output:0*
T0*A
_output_shapes/
-:+                           @2

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+                           @:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
к
╖
+__inference_Generator_layer_call_fn_1173087

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_Generator_layer_call_and_return_conditional_losses_11725092
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:         d::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╓
п
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1173407

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%═╠L>2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1з
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ш
Й
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1173223

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
А
E
)__inference_dropout_layer_call_fn_1173276

inputs
identityс
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_11721952
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╓
п
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1171981

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%═╠L>2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1з
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
д
к
7__inference_batch_normalization_1_layer_call_fn_1173337

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11718332
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
И
G
+__inference_dropout_2_layer_call_fn_1173478

inputs
identityф
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_11723612
PartitionedCallЗ
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           А:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
ж
к
7__inference_batch_normalization_1_layer_call_fn_1173350

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall╗
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11718642
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
┼
{
&__inference_cov3_layer_call_fn_1172072

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_cov3_layer_call_and_return_conditional_losses_11720622
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,                           А::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
ж
Л
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1173425

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А:::::j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╩
п
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1171833

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%═╠L>2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Л
b
)__inference_dropout_layer_call_fn_1173271

inputs
identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_11721902
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
░
╖
+__inference_Generator_layer_call_fn_1173136

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityИвStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_Generator_layer_call_and_return_conditional_losses_11726212
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:         d::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
м
f
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_1173281

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           @2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs"╕L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЯ
?
	gen_noise2
serving_default_gen_noise:0         d@
cov38
StatefulPartitionedCall:0         mYtensorflow/serving/predict:─┌
Ь|
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
	variables
trainable_variables
regularization_losses
	keras_api

signatures
+╗&call_and_return_all_conditional_losses
╝_default_save_signature
╜__call__"╖w
_tf_keras_networkЫw{"class_name": "Functional", "name": "Generator", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Generator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gen_noise"}, "name": "gen_noise", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 429, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["gen_noise", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [13, 11, 3]}}, "name": "reshape", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr2", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu", "inbound_nodes": [[["covtr2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr3", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_1", "inbound_nodes": [[["covtr3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr4", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_2", "inbound_nodes": [[["covtr4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "cov3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "cov3", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}], "input_layers": [["gen_noise", 0, 0]], "output_layers": [["cov3", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Generator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gen_noise"}, "name": "gen_noise", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 429, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["gen_noise", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [13, 11, 3]}}, "name": "reshape", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr2", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu", "inbound_nodes": [[["covtr2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr3", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_1", "inbound_nodes": [[["covtr3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr4", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_2", "inbound_nodes": [[["covtr4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "cov3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "cov3", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}], "input_layers": [["gen_noise", 0, 0]], "output_layers": [["cov3", 0, 0]]}}}
ё"ю
_tf_keras_input_layer╬{"class_name": "InputLayer", "name": "gen_noise", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gen_noise"}}
ё

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
+╛&call_and_return_all_conditional_losses
┐__call__"╩
_tf_keras_layer░{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 429, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
ў
	variables
trainable_variables
regularization_losses
	keras_api
+└&call_and_return_all_conditional_losses
┴__call__"ц
_tf_keras_layer╠{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [13, 11, 3]}}}
н


 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
+┬&call_and_return_all_conditional_losses
├__call__"Ж	
_tf_keras_layerь{"class_name": "Conv2DTranspose", "name": "covtr2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 11, 3]}}
▄
&	variables
'trainable_variables
(regularization_losses
)	keras_api
+─&call_and_return_all_conditional_losses
┼__call__"╦
_tf_keras_layer▒{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
╖	
*axis
	+gamma
,beta
-moving_mean
.moving_variance
/	variables
0trainable_variables
1regularization_losses
2	keras_api
+╞&call_and_return_all_conditional_losses
╟__call__"с
_tf_keras_layer╟{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 22, 32]}}
у
3	variables
4trainable_variables
5regularization_losses
6	keras_api
+╚&call_and_return_all_conditional_losses
╔__call__"╥
_tf_keras_layer╕{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
п


7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
+╩&call_and_return_all_conditional_losses
╦__call__"И	
_tf_keras_layerю{"class_name": "Conv2DTranspose", "name": "covtr3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 22, 32]}}
р
=	variables
>trainable_variables
?regularization_losses
@	keras_api
+╠&call_and_return_all_conditional_losses
═__call__"╧
_tf_keras_layer╡{"class_name": "LeakyReLU", "name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
╗	
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
+╬&call_and_return_all_conditional_losses
╧__call__"х
_tf_keras_layer╦{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 52, 44, 64]}}
ч
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
+╨&call_and_return_all_conditional_losses
╤__call__"╓
_tf_keras_layer╝{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
░


Nkernel
Obias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
+╥&call_and_return_all_conditional_losses
╙__call__"Й	
_tf_keras_layerя{"class_name": "Conv2DTranspose", "name": "covtr4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 52, 44, 64]}}
р
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
+╘&call_and_return_all_conditional_losses
╒__call__"╧
_tf_keras_layer╡{"class_name": "LeakyReLU", "name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
╛	
Xaxis
	Ygamma
Zbeta
[moving_mean
\moving_variance
]	variables
^trainable_variables
_regularization_losses
`	keras_api
+╓&call_and_return_all_conditional_losses
╫__call__"ш
_tf_keras_layer╬{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 104, 88, 128]}}
ч
a	variables
btrainable_variables
cregularization_losses
d	keras_api
+╪&call_and_return_all_conditional_losses
┘__call__"╓
_tf_keras_layer╝{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
п


ekernel
fbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
+┌&call_and_return_all_conditional_losses
█__call__"И	
_tf_keras_layerю{"class_name": "Conv2DTranspose", "name": "cov3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "cov3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 104, 88, 128]}}
╞
0
1
 2
!3
+4
,5
-6
.7
78
89
B10
C11
D12
E13
N14
O15
Y16
Z17
[18
\19
e20
f21"
trackable_list_wrapper
Ц
0
1
 2
!3
+4
,5
76
87
B8
C9
N10
O11
Y12
Z13
e14
f15"
trackable_list_wrapper
 "
trackable_list_wrapper
╬
	variables
klayer_metrics

llayers
trainable_variables
mnon_trainable_variables
nlayer_regularization_losses
ometrics
regularization_losses
╜__call__
╝_default_save_signature
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses"
_generic_user_object
-
▄serving_default"
signature_map
:	dн2dense/kernel
:н2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
░
	variables
player_metrics

qlayers
trainable_variables
rnon_trainable_variables
slayer_regularization_losses
tmetrics
regularization_losses
┐__call__
+╛&call_and_return_all_conditional_losses
'╛"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
	variables
ulayer_metrics

vlayers
trainable_variables
wnon_trainable_variables
xlayer_regularization_losses
ymetrics
regularization_losses
┴__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
':% 2covtr2/kernel
: 2covtr2/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
░
"	variables
zlayer_metrics

{layers
#trainable_variables
|non_trainable_variables
}layer_regularization_losses
~metrics
$regularization_losses
├__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
┤
&	variables
layer_metrics
Аlayers
'trainable_variables
Бnon_trainable_variables
 Вlayer_regularization_losses
Гmetrics
(regularization_losses
┼__call__
+─&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':% 2batch_normalization/gamma
&:$ 2batch_normalization/beta
/:-  (2batch_normalization/moving_mean
3:1  (2#batch_normalization/moving_variance
<
+0
,1
-2
.3"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
/	variables
Дlayer_metrics
Еlayers
0trainable_variables
Жnon_trainable_variables
 Зlayer_regularization_losses
Иmetrics
1regularization_losses
╟__call__
+╞&call_and_return_all_conditional_losses
'╞"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
3	variables
Йlayer_metrics
Кlayers
4trainable_variables
Лnon_trainable_variables
 Мlayer_regularization_losses
Нmetrics
5regularization_losses
╔__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
':%@ 2covtr3/kernel
:@2covtr3/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
9	variables
Оlayer_metrics
Пlayers
:trainable_variables
Рnon_trainable_variables
 Сlayer_regularization_losses
Тmetrics
;regularization_losses
╦__call__
+╩&call_and_return_all_conditional_losses
'╩"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
=	variables
Уlayer_metrics
Фlayers
>trainable_variables
Хnon_trainable_variables
 Цlayer_regularization_losses
Чmetrics
?regularization_losses
═__call__
+╠&call_and_return_all_conditional_losses
'╠"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
<
B0
C1
D2
E3"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
F	variables
Шlayer_metrics
Щlayers
Gtrainable_variables
Ъnon_trainable_variables
 Ыlayer_regularization_losses
Ьmetrics
Hregularization_losses
╧__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
J	variables
Эlayer_metrics
Юlayers
Ktrainable_variables
Яnon_trainable_variables
 аlayer_regularization_losses
бmetrics
Lregularization_losses
╤__call__
+╨&call_and_return_all_conditional_losses
'╨"call_and_return_conditional_losses"
_generic_user_object
(:&А@2covtr4/kernel
:А2covtr4/bias
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
P	variables
вlayer_metrics
гlayers
Qtrainable_variables
дnon_trainable_variables
 еlayer_regularization_losses
жmetrics
Rregularization_losses
╙__call__
+╥&call_and_return_all_conditional_losses
'╥"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
T	variables
зlayer_metrics
иlayers
Utrainable_variables
йnon_trainable_variables
 кlayer_regularization_losses
лmetrics
Vregularization_losses
╒__call__
+╘&call_and_return_all_conditional_losses
'╘"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(А2batch_normalization_2/gamma
):'А2batch_normalization_2/beta
2:0А (2!batch_normalization_2/moving_mean
6:4А (2%batch_normalization_2/moving_variance
<
Y0
Z1
[2
\3"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
]	variables
мlayer_metrics
нlayers
^trainable_variables
оnon_trainable_variables
 пlayer_regularization_losses
░metrics
_regularization_losses
╫__call__
+╓&call_and_return_all_conditional_losses
'╓"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
a	variables
▒layer_metrics
▓layers
btrainable_variables
│non_trainable_variables
 ┤layer_regularization_losses
╡metrics
cregularization_losses
┘__call__
+╪&call_and_return_all_conditional_losses
'╪"call_and_return_conditional_losses"
_generic_user_object
&:$А2cov3/kernel
:2	cov3/bias
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
g	variables
╢layer_metrics
╖layers
htrainable_variables
╕non_trainable_variables
 ╣layer_regularization_losses
║metrics
iregularization_losses
█__call__
+┌&call_and_return_all_conditional_losses
'┌"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
Ц
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
J
-0
.1
D2
E3
[4
\5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ц2у
F__inference_Generator_layer_call_and_return_conditional_losses_1173038
F__inference_Generator_layer_call_and_return_conditional_losses_1172892
F__inference_Generator_layer_call_and_return_conditional_losses_1172380
F__inference_Generator_layer_call_and_return_conditional_losses_1172443└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
т2▀
"__inference__wrapped_model_1171579╕
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *(в%
#К 
	gen_noise         d
·2ў
+__inference_Generator_layer_call_fn_1173087
+__inference_Generator_layer_call_fn_1173136
+__inference_Generator_layer_call_fn_1172556
+__inference_Generator_layer_call_fn_1172668└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ь2щ
B__inference_dense_layer_call_and_return_conditional_losses_1173147в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_dense_layer_call_fn_1173156в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_reshape_layer_call_and_return_conditional_losses_1173170в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙2╨
)__inference_reshape_layer_call_fn_1173175в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
в2Я
C__inference_covtr2_layer_call_and_return_conditional_losses_1171613╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
З2Д
(__inference_covtr2_layer_call_fn_1171623╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
Є2я
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_1173180в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╫2╘
-__inference_leaky_re_lu_layer_call_fn_1173185в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▐2█
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1173223
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1173205┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2е
5__inference_batch_normalization_layer_call_fn_1173236
5__inference_batch_normalization_layer_call_fn_1173249┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╞2├
D__inference_dropout_layer_call_and_return_conditional_losses_1173261
D__inference_dropout_layer_call_and_return_conditional_losses_1173266┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Р2Н
)__inference_dropout_layer_call_fn_1173271
)__inference_dropout_layer_call_fn_1173276┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
в2Я
C__inference_covtr3_layer_call_and_return_conditional_losses_1171761╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
З2Д
(__inference_covtr3_layer_call_fn_1171771╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
Ї2ё
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_1173281в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┘2╓
/__inference_leaky_re_lu_1_layer_call_fn_1173286в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
т2▀
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1173324
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1173306┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
м2й
7__inference_batch_normalization_1_layer_call_fn_1173337
7__inference_batch_normalization_1_layer_call_fn_1173350┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╩2╟
F__inference_dropout_1_layer_call_and_return_conditional_losses_1173367
F__inference_dropout_1_layer_call_and_return_conditional_losses_1173362┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ф2С
+__inference_dropout_1_layer_call_fn_1173372
+__inference_dropout_1_layer_call_fn_1173377┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
в2Я
C__inference_covtr4_layer_call_and_return_conditional_losses_1171909╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
З2Д
(__inference_covtr4_layer_call_fn_1171919╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
Ї2ё
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_1173382в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┘2╓
/__inference_leaky_re_lu_2_layer_call_fn_1173387в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
т2▀
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1173407
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1173425┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
м2й
7__inference_batch_normalization_2_layer_call_fn_1173438
7__inference_batch_normalization_2_layer_call_fn_1173451┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╩2╟
F__inference_dropout_2_layer_call_and_return_conditional_losses_1173468
F__inference_dropout_2_layer_call_and_return_conditional_losses_1173463┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ф2С
+__inference_dropout_2_layer_call_fn_1173473
+__inference_dropout_2_layer_call_fn_1173478┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
б2Ю
A__inference_cov3_layer_call_and_return_conditional_losses_1172062╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
Ж2Г
&__inference_cov3_layer_call_fn_1172072╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
6B4
%__inference_signature_wrapper_1172719	gen_noiseр
F__inference_Generator_layer_call_and_return_conditional_losses_1172380Х !+,-.78BCDENOYZ[\ef:в7
0в-
#К 
	gen_noise         d
p

 
к "?в<
5К2
0+                           
Ъ р
F__inference_Generator_layer_call_and_return_conditional_losses_1172443Х !+,-.78BCDENOYZ[\ef:в7
0в-
#К 
	gen_noise         d
p 

 
к "?в<
5К2
0+                           
Ъ ╦
F__inference_Generator_layer_call_and_return_conditional_losses_1172892А !+,-.78BCDENOYZ[\ef7в4
-в*
 К
inputs         d
p

 
к "-в*
#К 
0         mY
Ъ ╦
F__inference_Generator_layer_call_and_return_conditional_losses_1173038А !+,-.78BCDENOYZ[\ef7в4
-в*
 К
inputs         d
p 

 
к "-в*
#К 
0         mY
Ъ ╕
+__inference_Generator_layer_call_fn_1172556И !+,-.78BCDENOYZ[\ef:в7
0в-
#К 
	gen_noise         d
p

 
к "2К/+                           ╕
+__inference_Generator_layer_call_fn_1172668И !+,-.78BCDENOYZ[\ef:в7
0в-
#К 
	gen_noise         d
p 

 
к "2К/+                           ╡
+__inference_Generator_layer_call_fn_1173087Е !+,-.78BCDENOYZ[\ef7в4
-в*
 К
inputs         d
p

 
к "2К/+                           ╡
+__inference_Generator_layer_call_fn_1173136Е !+,-.78BCDENOYZ[\ef7в4
-в*
 К
inputs         d
p 

 
к "2К/+                           и
"__inference__wrapped_model_1171579Б !+,-.78BCDENOYZ[\ef2в/
(в%
#К 
	gen_noise         d
к "3к0
.
cov3&К#
cov3         mYэ
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1173306ЦBCDEMвJ
Cв@
:К7
inputs+                           @
p
к "?в<
5К2
0+                           @
Ъ э
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1173324ЦBCDEMвJ
Cв@
:К7
inputs+                           @
p 
к "?в<
5К2
0+                           @
Ъ ┼
7__inference_batch_normalization_1_layer_call_fn_1173337ЙBCDEMвJ
Cв@
:К7
inputs+                           @
p
к "2К/+                           @┼
7__inference_batch_normalization_1_layer_call_fn_1173350ЙBCDEMвJ
Cв@
:К7
inputs+                           @
p 
к "2К/+                           @я
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1173407ШYZ[\NвK
DвA
;К8
inputs,                           А
p
к "@в=
6К3
0,                           А
Ъ я
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1173425ШYZ[\NвK
DвA
;К8
inputs,                           А
p 
к "@в=
6К3
0,                           А
Ъ ╟
7__inference_batch_normalization_2_layer_call_fn_1173438ЛYZ[\NвK
DвA
;К8
inputs,                           А
p
к "3К0,                           А╟
7__inference_batch_normalization_2_layer_call_fn_1173451ЛYZ[\NвK
DвA
;К8
inputs,                           А
p 
к "3К0,                           Аы
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1173205Ц+,-.MвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ ы
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1173223Ц+,-.MвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ ├
5__inference_batch_normalization_layer_call_fn_1173236Й+,-.MвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            ├
5__inference_batch_normalization_layer_call_fn_1173249Й+,-.MвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            ╫
A__inference_cov3_layer_call_and_return_conditional_losses_1172062СefJвG
@в=
;К8
inputs,                           А
к "?в<
5К2
0+                           
Ъ п
&__inference_cov3_layer_call_fn_1172072ДefJвG
@в=
;К8
inputs,                           А
к "2К/+                           ╪
C__inference_covtr2_layer_call_and_return_conditional_losses_1171613Р !IвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                            
Ъ ░
(__inference_covtr2_layer_call_fn_1171623Г !IвF
?в<
:К7
inputs+                           
к "2К/+                            ╪
C__inference_covtr3_layer_call_and_return_conditional_losses_1171761Р78IвF
?в<
:К7
inputs+                            
к "?в<
5К2
0+                           @
Ъ ░
(__inference_covtr3_layer_call_fn_1171771Г78IвF
?в<
:К7
inputs+                            
к "2К/+                           @┘
C__inference_covtr4_layer_call_and_return_conditional_losses_1171909СNOIвF
?в<
:К7
inputs+                           @
к "@в=
6К3
0,                           А
Ъ ▒
(__inference_covtr4_layer_call_fn_1171919ДNOIвF
?в<
:К7
inputs+                           @
к "3К0,                           Аг
B__inference_dense_layer_call_and_return_conditional_losses_1173147]/в,
%в"
 К
inputs         d
к "&в#
К
0         н
Ъ {
'__inference_dense_layer_call_fn_1173156P/в,
%в"
 К
inputs         d
к "К         н█
F__inference_dropout_1_layer_call_and_return_conditional_losses_1173362РMвJ
Cв@
:К7
inputs+                           @
p
к "?в<
5К2
0+                           @
Ъ █
F__inference_dropout_1_layer_call_and_return_conditional_losses_1173367РMвJ
Cв@
:К7
inputs+                           @
p 
к "?в<
5К2
0+                           @
Ъ │
+__inference_dropout_1_layer_call_fn_1173372ГMвJ
Cв@
:К7
inputs+                           @
p
к "2К/+                           @│
+__inference_dropout_1_layer_call_fn_1173377ГMвJ
Cв@
:К7
inputs+                           @
p 
к "2К/+                           @▌
F__inference_dropout_2_layer_call_and_return_conditional_losses_1173463ТNвK
DвA
;К8
inputs,                           А
p
к "@в=
6К3
0,                           А
Ъ ▌
F__inference_dropout_2_layer_call_and_return_conditional_losses_1173468ТNвK
DвA
;К8
inputs,                           А
p 
к "@в=
6К3
0,                           А
Ъ ╡
+__inference_dropout_2_layer_call_fn_1173473ЕNвK
DвA
;К8
inputs,                           А
p
к "3К0,                           А╡
+__inference_dropout_2_layer_call_fn_1173478ЕNвK
DвA
;К8
inputs,                           А
p 
к "3К0,                           А┘
D__inference_dropout_layer_call_and_return_conditional_losses_1173261РMвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ ┘
D__inference_dropout_layer_call_and_return_conditional_losses_1173266РMвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ ▒
)__inference_dropout_layer_call_fn_1173271ГMвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            ▒
)__inference_dropout_layer_call_fn_1173276ГMвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            █
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_1173281МIвF
?в<
:К7
inputs+                           @
к "?в<
5К2
0+                           @
Ъ ▓
/__inference_leaky_re_lu_1_layer_call_fn_1173286IвF
?в<
:К7
inputs+                           @
к "2К/+                           @▌
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_1173382ОJвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ╡
/__inference_leaky_re_lu_2_layer_call_fn_1173387БJвG
@в=
;К8
inputs,                           А
к "3К0,                           А┘
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_1173180МIвF
?в<
:К7
inputs+                            
к "?в<
5К2
0+                            
Ъ ░
-__inference_leaky_re_lu_layer_call_fn_1173185IвF
?в<
:К7
inputs+                            
к "2К/+                            й
D__inference_reshape_layer_call_and_return_conditional_losses_1173170a0в-
&в#
!К
inputs         н
к "-в*
#К 
0         
Ъ Б
)__inference_reshape_layer_call_fn_1173175T0в-
&в#
!К
inputs         н
к " К         ╕
%__inference_signature_wrapper_1172719О !+,-.78BCDENOYZ[\ef?в<
в 
5к2
0
	gen_noise#К 
	gen_noise         d"3к0
.
cov3&К#
cov3         mY