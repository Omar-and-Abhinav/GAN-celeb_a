ЂЬ
ЭЃ
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
dtypetype
О
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18уб
u
Dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d­*
shared_nameDense/kernel
n
 Dense/kernel/Read/ReadVariableOpReadVariableOpDense/kernel*
_output_shapes
:	d­*
dtype0
m

Dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:­*
shared_name
Dense/bias
f
Dense/bias/Read/ReadVariableOpReadVariableOp
Dense/bias*
_output_shapes	
:­*
dtype0
~
covtr3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr3/kernel
w
!covtr3/kernel/Read/ReadVariableOpReadVariableOpcovtr3/kernel*&
_output_shapes
:*
dtype0
n
covtr3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr3/bias
g
covtr3/bias/Read/ReadVariableOpReadVariableOpcovtr3/bias*
_output_shapes
:*
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
~
covtr4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr4/kernel
w
!covtr4/kernel/Read/ReadVariableOpReadVariableOpcovtr4/kernel*&
_output_shapes
:*
dtype0
n
covtr4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr4/bias
g
covtr4/bias/Read/ReadVariableOpReadVariableOpcovtr4/bias*
_output_shapes
:*
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:*
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:*
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
Ђ
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
~
covtr5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr5/kernel
w
!covtr5/kernel/Read/ReadVariableOpReadVariableOpcovtr5/kernel*&
_output_shapes
:*
dtype0
n
covtr5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr5/bias
g
covtr5/bias/Read/ReadVariableOpReadVariableOpcovtr5/bias*
_output_shapes
:*
dtype0

batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_2/gamma

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:*
dtype0

batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:*
dtype0

!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:*
dtype0
Ђ
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:*
dtype0
~
covtr6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namecovtr6/kernel
w
!covtr6/kernel/Read/ReadVariableOpReadVariableOpcovtr6/kernel*&
_output_shapes
:@*
dtype0
n
covtr6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namecovtr6/bias
g
covtr6/bias/Read/ReadVariableOpReadVariableOpcovtr6/bias*
_output_shapes
:@*
dtype0

batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_3/gamma

/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:@*
dtype0

batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_3/beta

.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:@*
dtype0

!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_3/moving_mean

5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
Ђ
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_3/moving_variance

9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0

covtr10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namecovtr10/kernel
y
"covtr10/kernel/Read/ReadVariableOpReadVariableOpcovtr10/kernel*&
_output_shapes
:@*
dtype0
p
covtr10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr10/bias
i
 covtr10/bias/Read/ReadVariableOpReadVariableOpcovtr10/bias*
_output_shapes
:*
dtype0

NoOpNoOp
ЎH
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*щG
valueпGBмG BеG
М
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
R
&trainable_variables
'regularization_losses
(	variables
)	keras_api

*axis
	+gamma
,beta
-moving_mean
.moving_variance
/trainable_variables
0regularization_losses
1	variables
2	keras_api
h

3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
R
9trainable_variables
:regularization_losses
;	variables
<	keras_api

=axis
	>gamma
?beta
@moving_mean
Amoving_variance
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
h

Fkernel
Gbias
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
R
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api

Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
h

Ykernel
Zbias
[trainable_variables
\regularization_losses
]	variables
^	keras_api
R
_trainable_variables
`regularization_losses
a	variables
b	keras_api

caxis
	dgamma
ebeta
fmoving_mean
gmoving_variance
htrainable_variables
iregularization_losses
j	variables
k	keras_api
h

lkernel
mbias
ntrainable_variables
oregularization_losses
p	variables
q	keras_api

0
1
 2
!3
+4
,5
36
47
>8
?9
F10
G11
Q12
R13
Y14
Z15
d16
e17
l18
m19
 
ж
0
1
 2
!3
+4
,5
-6
.7
38
49
>10
?11
@12
A13
F14
G15
Q16
R17
S18
T19
Y20
Z21
d22
e23
f24
g25
l26
m27
­

rlayers
slayer_regularization_losses
tlayer_metrics
unon_trainable_variables
vmetrics
trainable_variables
regularization_losses
	variables
 
XV
VARIABLE_VALUEDense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
Dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

wlayers
xlayer_regularization_losses
ylayer_metrics
znon_trainable_variables
{metrics
trainable_variables
regularization_losses
	variables
 
 
 
Ў

|layers
}layer_regularization_losses
~layer_metrics
non_trainable_variables
metrics
trainable_variables
regularization_losses
	variables
YW
VARIABLE_VALUEcovtr3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
В
layers
 layer_regularization_losses
layer_metrics
non_trainable_variables
metrics
"trainable_variables
#regularization_losses
$	variables
 
 
 
В
layers
 layer_regularization_losses
layer_metrics
non_trainable_variables
metrics
&trainable_variables
'regularization_losses
(	variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

+0
,1
 

+0
,1
-2
.3
В
layers
 layer_regularization_losses
layer_metrics
non_trainable_variables
metrics
/trainable_variables
0regularization_losses
1	variables
YW
VARIABLE_VALUEcovtr4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41
В
layers
 layer_regularization_losses
layer_metrics
non_trainable_variables
metrics
5trainable_variables
6regularization_losses
7	variables
 
 
 
В
layers
 layer_regularization_losses
layer_metrics
non_trainable_variables
metrics
9trainable_variables
:regularization_losses
;	variables
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

>0
?1
 

>0
?1
@2
A3
В
layers
 layer_regularization_losses
layer_metrics
non_trainable_variables
metrics
Btrainable_variables
Cregularization_losses
D	variables
YW
VARIABLE_VALUEcovtr5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

F0
G1
 

F0
G1
В
layers
  layer_regularization_losses
Ёlayer_metrics
Ђnon_trainable_variables
Ѓmetrics
Htrainable_variables
Iregularization_losses
J	variables
 
 
 
В
Єlayers
 Ѕlayer_regularization_losses
Іlayer_metrics
Їnon_trainable_variables
Јmetrics
Ltrainable_variables
Mregularization_losses
N	variables
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1
 

Q0
R1
S2
T3
В
Љlayers
 Њlayer_regularization_losses
Ћlayer_metrics
Ќnon_trainable_variables
­metrics
Utrainable_variables
Vregularization_losses
W	variables
YW
VARIABLE_VALUEcovtr6/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr6/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1
 

Y0
Z1
В
Ўlayers
 Џlayer_regularization_losses
Аlayer_metrics
Бnon_trainable_variables
Вmetrics
[trainable_variables
\regularization_losses
]	variables
 
 
 
В
Гlayers
 Дlayer_regularization_losses
Еlayer_metrics
Жnon_trainable_variables
Зmetrics
_trainable_variables
`regularization_losses
a	variables
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

d0
e1
 

d0
e1
f2
g3
В
Иlayers
 Йlayer_regularization_losses
Кlayer_metrics
Лnon_trainable_variables
Мmetrics
htrainable_variables
iregularization_losses
j	variables
ZX
VARIABLE_VALUEcovtr10/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEcovtr10/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

l0
m1
 

l0
m1
В
Нlayers
 Оlayer_regularization_losses
Пlayer_metrics
Рnon_trainable_variables
Сmetrics
ntrainable_variables
oregularization_losses
p	variables
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
 
 
8
-0
.1
@2
A3
S4
T5
f6
g7
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

@0
A1
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
S0
T1
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
f0
g1
 
 
 
 
 
 
|
serving_default_gen_noisePlaceholder*'
_output_shapes
:џџџџџџџџџd*
dtype0*
shape:џџџџџџџџџd
Є
StatefulPartitionedCallStatefulPartitionedCallserving_default_gen_noiseDense/kernel
Dense/biascovtr3/kernelcovtr3/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancecovtr4/kernelcovtr4/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancecovtr5/kernelcovtr5/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancecovtr6/kernelcovtr6/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancecovtr10/kernelcovtr10/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџmY*>
_read_only_resource_inputs 
	
*2
config_proto" 

CPU

GPU2 *0J 8 *,
f'R%
#__inference_signature_wrapper_91789
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename Dense/kernel/Read/ReadVariableOpDense/bias/Read/ReadVariableOp!covtr3/kernel/Read/ReadVariableOpcovtr3/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp!covtr4/kernel/Read/ReadVariableOpcovtr4/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp!covtr5/kernel/Read/ReadVariableOpcovtr5/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp!covtr6/kernel/Read/ReadVariableOpcovtr6/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp"covtr10/kernel/Read/ReadVariableOp covtr10/bias/Read/ReadVariableOpConst*)
Tin"
 2*
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
GPU2 *0J 8 *'
f"R 
__inference__traced_save_92715
т
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameDense/kernel
Dense/biascovtr3/kernelcovtr3/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancecovtr4/kernelcovtr4/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancecovtr5/kernelcovtr5/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancecovtr6/kernelcovtr6/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancecovtr10/kernelcovtr10/bias*(
Tin!
2*
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
GPU2 *0J 8 **
f%R#
!__inference__traced_restore_92809УА
ђ!
Г
A__inference_covtr4_layer_call_and_return_conditional_losses_90644

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityD
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
value	B :2	
stack/3
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
strided_slice_3Г
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp№
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЄ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ЅA
ћ
__inference__traced_save_92715
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop,
(savev2_covtr3_kernel_read_readvariableop*
&savev2_covtr3_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop,
(savev2_covtr4_kernel_read_readvariableop*
&savev2_covtr4_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop,
(savev2_covtr5_kernel_read_readvariableop*
&savev2_covtr5_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop,
(savev2_covtr6_kernel_read_readvariableop*
&savev2_covtr6_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop-
)savev2_covtr10_kernel_read_readvariableop+
'savev2_covtr10_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_aa565e93f1e14dbfa6c25eab6ce087d7/part2	
Const_1
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameс
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ѓ
valueщBцB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesТ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesі
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop(savev2_covtr3_kernel_read_readvariableop&savev2_covtr3_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop(savev2_covtr4_kernel_read_readvariableop&savev2_covtr4_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop(savev2_covtr5_kernel_read_readvariableop&savev2_covtr5_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop(savev2_covtr6_kernel_read_readvariableop&savev2_covtr6_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop)savev2_covtr10_kernel_read_readvariableop'savev2_covtr10_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *+
dtypes!
22
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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

identity_1Identity_1:output:0*
_input_shapesё
ю: :	d­:­:::::::::::::::::::@:@:@:@:@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	d­:!

_output_shapes	
:­:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
::

_output_shapes
: 
Ш
­
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_91016

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%ЭЬL>2
FusedBatchNormV3џ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Њ
d
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_91276

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ѕM
Ч	
D__inference_Generator_layer_call_and_return_conditional_losses_91667

inputs
dense_91595
dense_91597
covtr3_91601
covtr3_91603
batch_normalization_91607
batch_normalization_91609
batch_normalization_91611
batch_normalization_91613
covtr4_91616
covtr4_91618
batch_normalization_1_91622
batch_normalization_1_91624
batch_normalization_1_91626
batch_normalization_1_91628
covtr5_91631
covtr5_91633
batch_normalization_2_91637
batch_normalization_2_91639
batch_normalization_2_91641
batch_normalization_2_91643
covtr6_91646
covtr6_91648
batch_normalization_3_91652
batch_normalization_3_91654
batch_normalization_3_91656
batch_normalization_3_91658
covtr10_91661
covtr10_91663
identityЂDense/StatefulPartitionedCallЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂcovtr10/StatefulPartitionedCallЂcovtr3/StatefulPartitionedCallЂcovtr4/StatefulPartitionedCallЂcovtr5/StatefulPartitionedCallЂcovtr6/StatefulPartitionedCall
Dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_91595dense_91597*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ­*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_Dense_layer_call_and_return_conditional_losses_911222
Dense/StatefulPartitionedCall§
reshape/PartitionedCallPartitionedCall&Dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_911522
reshape/PartitionedCallР
covtr3/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr3_91601covtr3_91603*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr3_layer_call_and_return_conditional_losses_904962 
covtr3/StatefulPartitionedCall
leaky_re_lu/PartitionedCallPartitionedCall'covtr3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_911702
leaky_re_lu/PartitionedCallП
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_91607batch_normalization_91609batch_normalization_91611batch_normalization_91613*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_905992-
+batch_normalization/StatefulPartitionedCallд
covtr4/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr4_91616covtr4_91618*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr4_layer_call_and_return_conditional_losses_906442 
covtr4/StatefulPartitionedCallЂ
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_912232
leaky_re_lu_1/PartitionedCallЯ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_91622batch_normalization_1_91624batch_normalization_1_91626batch_normalization_1_91628*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_907472/
-batch_normalization_1/StatefulPartitionedCallж
covtr5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr5_91631covtr5_91633*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr5_layer_call_and_return_conditional_losses_907922 
covtr5/StatefulPartitionedCallЂ
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_912762
leaky_re_lu_2/PartitionedCallЯ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_91637batch_normalization_2_91639batch_normalization_2_91641batch_normalization_2_91643*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_908952/
-batch_normalization_2/StatefulPartitionedCallж
covtr6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0covtr6_91646covtr6_91648*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr6_layer_call_and_return_conditional_losses_909442 
covtr6/StatefulPartitionedCallЂ
leaky_re_lu_3/PartitionedCallPartitionedCall'covtr6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_913292
leaky_re_lu_3/PartitionedCallЯ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0batch_normalization_3_91652batch_normalization_3_91654batch_normalization_3_91656batch_normalization_3_91658*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_910472/
-batch_normalization_3/StatefulPartitionedCallл
covtr10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0covtr10_91661covtr10_91663*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_covtr10_layer_call_and_return_conditional_losses_910972!
covtr10/StatefulPartitionedCall
IdentityIdentity(covtr10/StatefulPartitionedCall:output:0^Dense/StatefulPartitionedCall,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^covtr10/StatefulPartitionedCall^covtr3/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^covtr5/StatefulPartitionedCall^covtr6/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd::::::::::::::::::::::::::::2>
Dense/StatefulPartitionedCallDense/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
covtr10/StatefulPartitionedCallcovtr10/StatefulPartitionedCall2@
covtr3/StatefulPartitionedCallcovtr3/StatefulPartitionedCall2@
covtr4/StatefulPartitionedCallcovtr4/StatefulPartitionedCall2@
covtr5/StatefulPartitionedCallcovtr5/StatefulPartitionedCall2@
covtr6/StatefulPartitionedCallcovtr6/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Њ
d
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_92391

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Њ
d
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_92539

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Ћ
Ј
@__inference_Dense_layer_call_and_return_conditional_losses_92284

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d­*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ­2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:­*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ­2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ­2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ­2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџd:::O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ѕ
C
'__inference_reshape_layer_call_fn_92312

inputs
identityЭ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_911522
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ­:P L
(
_output_shapes
:џџџџџџџџџ­
 
_user_specified_nameinputs
 
Ј
5__inference_batch_normalization_1_layer_call_fn_92447

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_907162
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ш
­
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_92490

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%ЭЬL>2
FusedBatchNormV3џ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

I
-__inference_leaky_re_lu_2_layer_call_fn_92470

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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_912762
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Х
|
'__inference_covtr10_layer_call_fn_91107

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_covtr10_layer_call_and_return_conditional_losses_910972
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
ћ

)__inference_Generator_layer_call_fn_92273

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

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identityЂStatefulPartitionedCallі
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*>
_read_only_resource_inputs 
	
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_Generator_layer_call_and_return_conditional_losses_916672
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
ўM
Ъ	
D__inference_Generator_layer_call_and_return_conditional_losses_91453
	gen_noise
dense_91381
dense_91383
covtr3_91387
covtr3_91389
batch_normalization_91393
batch_normalization_91395
batch_normalization_91397
batch_normalization_91399
covtr4_91402
covtr4_91404
batch_normalization_1_91408
batch_normalization_1_91410
batch_normalization_1_91412
batch_normalization_1_91414
covtr5_91417
covtr5_91419
batch_normalization_2_91423
batch_normalization_2_91425
batch_normalization_2_91427
batch_normalization_2_91429
covtr6_91432
covtr6_91434
batch_normalization_3_91438
batch_normalization_3_91440
batch_normalization_3_91442
batch_normalization_3_91444
covtr10_91447
covtr10_91449
identityЂDense/StatefulPartitionedCallЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂcovtr10/StatefulPartitionedCallЂcovtr3/StatefulPartitionedCallЂcovtr4/StatefulPartitionedCallЂcovtr5/StatefulPartitionedCallЂcovtr6/StatefulPartitionedCall
Dense/StatefulPartitionedCallStatefulPartitionedCall	gen_noisedense_91381dense_91383*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ­*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_Dense_layer_call_and_return_conditional_losses_911222
Dense/StatefulPartitionedCall§
reshape/PartitionedCallPartitionedCall&Dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_911522
reshape/PartitionedCallР
covtr3/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr3_91387covtr3_91389*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr3_layer_call_and_return_conditional_losses_904962 
covtr3/StatefulPartitionedCall
leaky_re_lu/PartitionedCallPartitionedCall'covtr3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_911702
leaky_re_lu/PartitionedCallП
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_91393batch_normalization_91395batch_normalization_91397batch_normalization_91399*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_905992-
+batch_normalization/StatefulPartitionedCallд
covtr4/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr4_91402covtr4_91404*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr4_layer_call_and_return_conditional_losses_906442 
covtr4/StatefulPartitionedCallЂ
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_912232
leaky_re_lu_1/PartitionedCallЯ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_91408batch_normalization_1_91410batch_normalization_1_91412batch_normalization_1_91414*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_907472/
-batch_normalization_1/StatefulPartitionedCallж
covtr5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr5_91417covtr5_91419*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr5_layer_call_and_return_conditional_losses_907922 
covtr5/StatefulPartitionedCallЂ
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_912762
leaky_re_lu_2/PartitionedCallЯ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_91423batch_normalization_2_91425batch_normalization_2_91427batch_normalization_2_91429*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_908952/
-batch_normalization_2/StatefulPartitionedCallж
covtr6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0covtr6_91432covtr6_91434*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr6_layer_call_and_return_conditional_losses_909442 
covtr6/StatefulPartitionedCallЂ
leaky_re_lu_3/PartitionedCallPartitionedCall'covtr6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_913292
leaky_re_lu_3/PartitionedCallЯ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0batch_normalization_3_91438batch_normalization_3_91440batch_normalization_3_91442batch_normalization_3_91444*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_910472/
-batch_normalization_3/StatefulPartitionedCallл
covtr10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0covtr10_91447covtr10_91449*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_covtr10_layer_call_and_return_conditional_losses_910972!
covtr10/StatefulPartitionedCall
IdentityIdentity(covtr10/StatefulPartitionedCall:output:0^Dense/StatefulPartitionedCall,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^covtr10/StatefulPartitionedCall^covtr3/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^covtr5/StatefulPartitionedCall^covtr6/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd::::::::::::::::::::::::::::2>
Dense/StatefulPartitionedCallDense/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
covtr10/StatefulPartitionedCallcovtr10/StatefulPartitionedCall2@
covtr3/StatefulPartitionedCallcovtr3/StatefulPartitionedCall2@
covtr4/StatefulPartitionedCallcovtr4/StatefulPartitionedCall2@
covtr5/StatefulPartitionedCallcovtr5/StatefulPartitionedCall2@
covtr6/StatefulPartitionedCallcovtr6/StatefulPartitionedCall:R N
'
_output_shapes
:џџџџџџџџџd
#
_user_specified_name	gen_noise

I
-__inference_leaky_re_lu_3_layer_call_fn_92544

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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_913292
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs

I
-__inference_leaky_re_lu_1_layer_call_fn_92396

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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_912232
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
У
{
&__inference_covtr6_layer_call_fn_90954

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr6_layer_call_and_return_conditional_losses_909442
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ш
­
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_92564

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%ЭЬL>2
FusedBatchNormV3џ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Њ
d
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_91223

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ш
­
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_92416

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%ЭЬL>2
FusedBatchNormV3џ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_92434

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
Ј
@__inference_Dense_layer_call_and_return_conditional_losses_91122

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d­*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ­2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:­*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ­2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ­2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ­2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџd:::O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs


N__inference_batch_normalization_layer_call_and_return_conditional_losses_90599

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_92508

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ш
­
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_90864

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%ЭЬL>2
FusedBatchNormV3џ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

І
3__inference_batch_normalization_layer_call_fn_92373

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_905682
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ђ
Ј
5__inference_batch_normalization_1_layer_call_fn_92460

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_907472
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
П
п
D__inference_Generator_layer_call_and_return_conditional_losses_92151

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource3
/covtr3_conv2d_transpose_readvariableop_resource*
&covtr3_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource3
/covtr4_conv2d_transpose_readvariableop_resource*
&covtr4_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource3
/covtr5_conv2d_transpose_readvariableop_resource*
&covtr5_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource3
/covtr6_conv2d_transpose_readvariableop_resource*
&covtr6_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource4
0covtr10_conv2d_transpose_readvariableop_resource+
'covtr10_biasadd_readvariableop_resource
identity 
Dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	d­*
dtype02
Dense/MatMul/ReadVariableOp
Dense/MatMulMatMulinputs#Dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ­2
Dense/MatMul
Dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:­*
dtype02
Dense/BiasAdd/ReadVariableOp
Dense/BiasAddBiasAddDense/MatMul:product:0$Dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ­2
Dense/BiasAddk

Dense/ReluReluDense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ­2

Dense/Reluf
reshape/ShapeShapeDense/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
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
reshape/Reshape/shapeЁ
reshape/ReshapeReshapeDense/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
reshape/Reshaped
covtr3/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
covtr3/Shape
covtr3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr3/strided_slice/stack
covtr3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr3/strided_slice/stack_1
covtr3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr3/strided_slice/stack_2
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
value	B :2
covtr3/stack/1b
covtr3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
covtr3/stack/2b
covtr3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr3/stack/3М
covtr3/stackPackcovtr3/strided_slice:output:0covtr3/stack/1:output:0covtr3/stack/2:output:0covtr3/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr3/stack
covtr3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr3/strided_slice_1/stack
covtr3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr3/strided_slice_1/stack_1
covtr3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr3/strided_slice_1/stack_2
covtr3/strided_slice_1StridedSlicecovtr3/stack:output:0%covtr3/strided_slice_1/stack:output:0'covtr3/strided_slice_1/stack_1:output:0'covtr3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr3/strided_slice_1Ш
&covtr3/conv2d_transpose/ReadVariableOpReadVariableOp/covtr3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr3/conv2d_transpose/ReadVariableOp
covtr3/conv2d_transposeConv2DBackpropInputcovtr3/stack:output:0.covtr3/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
covtr3/conv2d_transposeЁ
covtr3/BiasAdd/ReadVariableOpReadVariableOp&covtr3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr3/BiasAdd/ReadVariableOpЎ
covtr3/BiasAddBiasAdd covtr3/conv2d_transpose:output:0%covtr3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
covtr3/BiasAdd
leaky_re_lu/LeakyRelu	LeakyRelucovtr3/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ2
leaky_re_lu/LeakyReluА
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOpЖ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1п
$batch_normalization/FusedBatchNormV3FusedBatchNormV3#leaky_re_lu/LeakyRelu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3t
covtr4/ShapeShape(batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr4/Shape
covtr4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr4/strided_slice/stack
covtr4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr4/strided_slice/stack_1
covtr4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr4/strided_slice/stack_2
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
value	B :42
covtr4/stack/1b
covtr4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :,2
covtr4/stack/2b
covtr4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr4/stack/3М
covtr4/stackPackcovtr4/strided_slice:output:0covtr4/stack/1:output:0covtr4/stack/2:output:0covtr4/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr4/stack
covtr4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr4/strided_slice_1/stack
covtr4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr4/strided_slice_1/stack_1
covtr4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr4/strided_slice_1/stack_2
covtr4/strided_slice_1StridedSlicecovtr4/stack:output:0%covtr4/strided_slice_1/stack:output:0'covtr4/strided_slice_1/stack_1:output:0'covtr4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr4/strided_slice_1Ш
&covtr4/conv2d_transpose/ReadVariableOpReadVariableOp/covtr4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr4/conv2d_transpose/ReadVariableOp
covtr4/conv2d_transposeConv2DBackpropInputcovtr4/stack:output:0.covtr4/conv2d_transpose/ReadVariableOp:value:0(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ4,*
paddingSAME*
strides
2
covtr4/conv2d_transposeЁ
covtr4/BiasAdd/ReadVariableOpReadVariableOp&covtr4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr4/BiasAdd/ReadVariableOpЎ
covtr4/BiasAddBiasAdd covtr4/conv2d_transpose:output:0%covtr4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ4,2
covtr4/BiasAdd
leaky_re_lu_1/LeakyRelu	LeakyRelucovtr4/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ4,2
leaky_re_lu_1/LeakyReluЖ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOpМ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1э
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_1/LeakyRelu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ4,:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3v
covtr5/ShapeShape*batch_normalization_1/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr5/Shape
covtr5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr5/strided_slice/stack
covtr5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr5/strided_slice/stack_1
covtr5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr5/strided_slice/stack_2
covtr5/strided_sliceStridedSlicecovtr5/Shape:output:0#covtr5/strided_slice/stack:output:0%covtr5/strided_slice/stack_1:output:0%covtr5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr5/strided_sliceb
covtr5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :h2
covtr5/stack/1b
covtr5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :X2
covtr5/stack/2b
covtr5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr5/stack/3М
covtr5/stackPackcovtr5/strided_slice:output:0covtr5/stack/1:output:0covtr5/stack/2:output:0covtr5/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr5/stack
covtr5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr5/strided_slice_1/stack
covtr5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr5/strided_slice_1/stack_1
covtr5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr5/strided_slice_1/stack_2
covtr5/strided_slice_1StridedSlicecovtr5/stack:output:0%covtr5/strided_slice_1/stack:output:0'covtr5/strided_slice_1/stack_1:output:0'covtr5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr5/strided_slice_1Ш
&covtr5/conv2d_transpose/ReadVariableOpReadVariableOp/covtr5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr5/conv2d_transpose/ReadVariableOp
covtr5/conv2d_transposeConv2DBackpropInputcovtr5/stack:output:0.covtr5/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџhX*
paddingSAME*
strides
2
covtr5/conv2d_transposeЁ
covtr5/BiasAdd/ReadVariableOpReadVariableOp&covtr5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr5/BiasAdd/ReadVariableOpЎ
covtr5/BiasAddBiasAdd covtr5/conv2d_transpose:output:0%covtr5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhX2
covtr5/BiasAdd
leaky_re_lu_2/LeakyRelu	LeakyRelucovtr5/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџhX2
leaky_re_lu_2/LeakyReluЖ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_2/ReadVariableOpМ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1щ
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1э
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_2/LeakyRelu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџhX:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3v
covtr6/ShapeShape*batch_normalization_2/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr6/Shape
covtr6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr6/strided_slice/stack
covtr6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr6/strided_slice/stack_1
covtr6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr6/strided_slice/stack_2
covtr6/strided_sliceStridedSlicecovtr6/Shape:output:0#covtr6/strided_slice/stack:output:0%covtr6/strided_slice/stack_1:output:0%covtr6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr6/strided_sliceb
covtr6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :k2
covtr6/stack/1b
covtr6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Y2
covtr6/stack/2b
covtr6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
covtr6/stack/3М
covtr6/stackPackcovtr6/strided_slice:output:0covtr6/stack/1:output:0covtr6/stack/2:output:0covtr6/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr6/stack
covtr6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr6/strided_slice_1/stack
covtr6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr6/strided_slice_1/stack_1
covtr6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr6/strided_slice_1/stack_2
covtr6/strided_slice_1StridedSlicecovtr6/stack:output:0%covtr6/strided_slice_1/stack:output:0'covtr6/strided_slice_1/stack_1:output:0'covtr6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr6/strided_slice_1Ш
&covtr6/conv2d_transpose/ReadVariableOpReadVariableOp/covtr6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&covtr6/conv2d_transpose/ReadVariableOp
covtr6/conv2d_transposeConv2DBackpropInputcovtr6/stack:output:0.covtr6/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџkY@*
paddingVALID*
strides
2
covtr6/conv2d_transposeЁ
covtr6/BiasAdd/ReadVariableOpReadVariableOp&covtr6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
covtr6/BiasAdd/ReadVariableOpЎ
covtr6/BiasAddBiasAdd covtr6/conv2d_transpose:output:0%covtr6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџkY@2
covtr6/BiasAdd
leaky_re_lu_3/LeakyRelu	LeakyRelucovtr6/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџkY@2
leaky_re_lu_3/LeakyReluЖ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOpМ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1щ
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1э
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_3/LeakyRelu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџkY@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3x
covtr10/ShapeShape*batch_normalization_3/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr10/Shape
covtr10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr10/strided_slice/stack
covtr10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr10/strided_slice/stack_1
covtr10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr10/strided_slice/stack_2
covtr10/strided_sliceStridedSlicecovtr10/Shape:output:0$covtr10/strided_slice/stack:output:0&covtr10/strided_slice/stack_1:output:0&covtr10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr10/strided_sliced
covtr10/stack/1Const*
_output_shapes
: *
dtype0*
value	B :m2
covtr10/stack/1d
covtr10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Y2
covtr10/stack/2d
covtr10/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr10/stack/3Т
covtr10/stackPackcovtr10/strided_slice:output:0covtr10/stack/1:output:0covtr10/stack/2:output:0covtr10/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr10/stack
covtr10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr10/strided_slice_1/stack
covtr10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr10/strided_slice_1/stack_1
covtr10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr10/strided_slice_1/stack_2
covtr10/strided_slice_1StridedSlicecovtr10/stack:output:0&covtr10/strided_slice_1/stack:output:0(covtr10/strided_slice_1/stack_1:output:0(covtr10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr10/strided_slice_1Ы
'covtr10/conv2d_transpose/ReadVariableOpReadVariableOp0covtr10_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02)
'covtr10/conv2d_transpose/ReadVariableOpЃ
covtr10/conv2d_transposeConv2DBackpropInputcovtr10/stack:output:0/covtr10/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџmY*
paddingVALID*
strides
2
covtr10/conv2d_transposeЄ
covtr10/BiasAdd/ReadVariableOpReadVariableOp'covtr10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
covtr10/BiasAdd/ReadVariableOpВ
covtr10/BiasAddBiasAdd!covtr10/conv2d_transpose:output:0&covtr10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџmY2
covtr10/BiasAdd
covtr10/SigmoidSigmoidcovtr10/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџmY2
covtr10/Sigmoido
IdentityIdentitycovtr10/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџmY2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd:::::::::::::::::::::::::::::O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
ђ!
Г
A__inference_covtr5_layer_call_and_return_conditional_losses_90792

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityD
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
value	B :2	
stack/3
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
strided_slice_3Г
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp№
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЄ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_92582

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
с
ж
 __inference__wrapped_model_90462
	gen_noise2
.generator_dense_matmul_readvariableop_resource3
/generator_dense_biasadd_readvariableop_resource=
9generator_covtr3_conv2d_transpose_readvariableop_resource4
0generator_covtr3_biasadd_readvariableop_resource9
5generator_batch_normalization_readvariableop_resource;
7generator_batch_normalization_readvariableop_1_resourceJ
Fgenerator_batch_normalization_fusedbatchnormv3_readvariableop_resourceL
Hgenerator_batch_normalization_fusedbatchnormv3_readvariableop_1_resource=
9generator_covtr4_conv2d_transpose_readvariableop_resource4
0generator_covtr4_biasadd_readvariableop_resource;
7generator_batch_normalization_1_readvariableop_resource=
9generator_batch_normalization_1_readvariableop_1_resourceL
Hgenerator_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceN
Jgenerator_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource=
9generator_covtr5_conv2d_transpose_readvariableop_resource4
0generator_covtr5_biasadd_readvariableop_resource;
7generator_batch_normalization_2_readvariableop_resource=
9generator_batch_normalization_2_readvariableop_1_resourceL
Hgenerator_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceN
Jgenerator_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource=
9generator_covtr6_conv2d_transpose_readvariableop_resource4
0generator_covtr6_biasadd_readvariableop_resource;
7generator_batch_normalization_3_readvariableop_resource=
9generator_batch_normalization_3_readvariableop_1_resourceL
Hgenerator_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceN
Jgenerator_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource>
:generator_covtr10_conv2d_transpose_readvariableop_resource5
1generator_covtr10_biasadd_readvariableop_resource
identityО
%Generator/Dense/MatMul/ReadVariableOpReadVariableOp.generator_dense_matmul_readvariableop_resource*
_output_shapes
:	d­*
dtype02'
%Generator/Dense/MatMul/ReadVariableOpЇ
Generator/Dense/MatMulMatMul	gen_noise-Generator/Dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ­2
Generator/Dense/MatMulН
&Generator/Dense/BiasAdd/ReadVariableOpReadVariableOp/generator_dense_biasadd_readvariableop_resource*
_output_shapes	
:­*
dtype02(
&Generator/Dense/BiasAdd/ReadVariableOpТ
Generator/Dense/BiasAddBiasAdd Generator/Dense/MatMul:product:0.Generator/Dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ­2
Generator/Dense/BiasAdd
Generator/Dense/ReluRelu Generator/Dense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ­2
Generator/Dense/Relu
Generator/reshape/ShapeShape"Generator/Dense/Relu:activations:0*
T0*
_output_shapes
:2
Generator/reshape/Shape
%Generator/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Generator/reshape/strided_slice/stack
'Generator/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/reshape/strided_slice/stack_1
'Generator/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/reshape/strided_slice/stack_2Ю
Generator/reshape/strided_sliceStridedSlice Generator/reshape/Shape:output:0.Generator/reshape/strided_slice/stack:output:00Generator/reshape/strided_slice/stack_1:output:00Generator/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
Generator/reshape/strided_slice
!Generator/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!Generator/reshape/Reshape/shape/1
!Generator/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!Generator/reshape/Reshape/shape/2
!Generator/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!Generator/reshape/Reshape/shape/3І
Generator/reshape/Reshape/shapePack(Generator/reshape/strided_slice:output:0*Generator/reshape/Reshape/shape/1:output:0*Generator/reshape/Reshape/shape/2:output:0*Generator/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
Generator/reshape/Reshape/shapeЩ
Generator/reshape/ReshapeReshape"Generator/Dense/Relu:activations:0(Generator/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
Generator/reshape/Reshape
Generator/covtr3/ShapeShape"Generator/reshape/Reshape:output:0*
T0*
_output_shapes
:2
Generator/covtr3/Shape
$Generator/covtr3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr3/strided_slice/stack
&Generator/covtr3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr3/strided_slice/stack_1
&Generator/covtr3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr3/strided_slice/stack_2Ш
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
value	B :2
Generator/covtr3/stack/1v
Generator/covtr3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr3/stack/2v
Generator/covtr3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr3/stack/3ј
Generator/covtr3/stackPack'Generator/covtr3/strided_slice:output:0!Generator/covtr3/stack/1:output:0!Generator/covtr3/stack/2:output:0!Generator/covtr3/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr3/stack
&Generator/covtr3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr3/strided_slice_1/stack
(Generator/covtr3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr3/strided_slice_1/stack_1
(Generator/covtr3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr3/strided_slice_1/stack_2в
 Generator/covtr3/strided_slice_1StridedSliceGenerator/covtr3/stack:output:0/Generator/covtr3/strided_slice_1/stack:output:01Generator/covtr3/strided_slice_1/stack_1:output:01Generator/covtr3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr3/strided_slice_1ц
0Generator/covtr3/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0Generator/covtr3/conv2d_transpose/ReadVariableOpО
!Generator/covtr3/conv2d_transposeConv2DBackpropInputGenerator/covtr3/stack:output:08Generator/covtr3/conv2d_transpose/ReadVariableOp:value:0"Generator/reshape/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2#
!Generator/covtr3/conv2d_transposeП
'Generator/covtr3/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Generator/covtr3/BiasAdd/ReadVariableOpж
Generator/covtr3/BiasAddBiasAdd*Generator/covtr3/conv2d_transpose:output:0/Generator/covtr3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
Generator/covtr3/BiasAddЃ
Generator/leaky_re_lu/LeakyRelu	LeakyRelu!Generator/covtr3/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ2!
Generator/leaky_re_lu/LeakyReluЮ
,Generator/batch_normalization/ReadVariableOpReadVariableOp5generator_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02.
,Generator/batch_normalization/ReadVariableOpд
.Generator/batch_normalization/ReadVariableOp_1ReadVariableOp7generator_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype020
.Generator/batch_normalization/ReadVariableOp_1
=Generator/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpFgenerator_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=Generator/batch_normalization/FusedBatchNormV3/ReadVariableOp
?Generator/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHgenerator_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?Generator/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ѕ
.Generator/batch_normalization/FusedBatchNormV3FusedBatchNormV3-Generator/leaky_re_lu/LeakyRelu:activations:04Generator/batch_normalization/ReadVariableOp:value:06Generator/batch_normalization/ReadVariableOp_1:value:0EGenerator/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0GGenerator/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 20
.Generator/batch_normalization/FusedBatchNormV3
Generator/covtr4/ShapeShape2Generator/batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr4/Shape
$Generator/covtr4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr4/strided_slice/stack
&Generator/covtr4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr4/strided_slice/stack_1
&Generator/covtr4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr4/strided_slice/stack_2Ш
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
value	B :42
Generator/covtr4/stack/1v
Generator/covtr4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :,2
Generator/covtr4/stack/2v
Generator/covtr4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr4/stack/3ј
Generator/covtr4/stackPack'Generator/covtr4/strided_slice:output:0!Generator/covtr4/stack/1:output:0!Generator/covtr4/stack/2:output:0!Generator/covtr4/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr4/stack
&Generator/covtr4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr4/strided_slice_1/stack
(Generator/covtr4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr4/strided_slice_1/stack_1
(Generator/covtr4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr4/strided_slice_1/stack_2в
 Generator/covtr4/strided_slice_1StridedSliceGenerator/covtr4/stack:output:0/Generator/covtr4/strided_slice_1/stack:output:01Generator/covtr4/strided_slice_1/stack_1:output:01Generator/covtr4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr4/strided_slice_1ц
0Generator/covtr4/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0Generator/covtr4/conv2d_transpose/ReadVariableOpЮ
!Generator/covtr4/conv2d_transposeConv2DBackpropInputGenerator/covtr4/stack:output:08Generator/covtr4/conv2d_transpose/ReadVariableOp:value:02Generator/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ4,*
paddingSAME*
strides
2#
!Generator/covtr4/conv2d_transposeП
'Generator/covtr4/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Generator/covtr4/BiasAdd/ReadVariableOpж
Generator/covtr4/BiasAddBiasAdd*Generator/covtr4/conv2d_transpose:output:0/Generator/covtr4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ4,2
Generator/covtr4/BiasAddЇ
!Generator/leaky_re_lu_1/LeakyRelu	LeakyRelu!Generator/covtr4/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ4,2#
!Generator/leaky_re_lu_1/LeakyReluд
.Generator/batch_normalization_1/ReadVariableOpReadVariableOp7generator_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype020
.Generator/batch_normalization_1/ReadVariableOpк
0Generator/batch_normalization_1/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype022
0Generator/batch_normalization_1/ReadVariableOp_1
?Generator/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?Generator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp
AGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
AGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Г
0Generator/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_1/LeakyRelu:activations:06Generator/batch_normalization_1/ReadVariableOp:value:08Generator/batch_normalization_1/ReadVariableOp_1:value:0GGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ4,:::::*
epsilon%o:*
is_training( 22
0Generator/batch_normalization_1/FusedBatchNormV3
Generator/covtr5/ShapeShape4Generator/batch_normalization_1/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr5/Shape
$Generator/covtr5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr5/strided_slice/stack
&Generator/covtr5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr5/strided_slice/stack_1
&Generator/covtr5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr5/strided_slice/stack_2Ш
Generator/covtr5/strided_sliceStridedSliceGenerator/covtr5/Shape:output:0-Generator/covtr5/strided_slice/stack:output:0/Generator/covtr5/strided_slice/stack_1:output:0/Generator/covtr5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Generator/covtr5/strided_slicev
Generator/covtr5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :h2
Generator/covtr5/stack/1v
Generator/covtr5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :X2
Generator/covtr5/stack/2v
Generator/covtr5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr5/stack/3ј
Generator/covtr5/stackPack'Generator/covtr5/strided_slice:output:0!Generator/covtr5/stack/1:output:0!Generator/covtr5/stack/2:output:0!Generator/covtr5/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr5/stack
&Generator/covtr5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr5/strided_slice_1/stack
(Generator/covtr5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr5/strided_slice_1/stack_1
(Generator/covtr5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr5/strided_slice_1/stack_2в
 Generator/covtr5/strided_slice_1StridedSliceGenerator/covtr5/stack:output:0/Generator/covtr5/strided_slice_1/stack:output:01Generator/covtr5/strided_slice_1/stack_1:output:01Generator/covtr5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr5/strided_slice_1ц
0Generator/covtr5/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0Generator/covtr5/conv2d_transpose/ReadVariableOpа
!Generator/covtr5/conv2d_transposeConv2DBackpropInputGenerator/covtr5/stack:output:08Generator/covtr5/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџhX*
paddingSAME*
strides
2#
!Generator/covtr5/conv2d_transposeП
'Generator/covtr5/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Generator/covtr5/BiasAdd/ReadVariableOpж
Generator/covtr5/BiasAddBiasAdd*Generator/covtr5/conv2d_transpose:output:0/Generator/covtr5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhX2
Generator/covtr5/BiasAddЇ
!Generator/leaky_re_lu_2/LeakyRelu	LeakyRelu!Generator/covtr5/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџhX2#
!Generator/leaky_re_lu_2/LeakyReluд
.Generator/batch_normalization_2/ReadVariableOpReadVariableOp7generator_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype020
.Generator/batch_normalization_2/ReadVariableOpк
0Generator/batch_normalization_2/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype022
0Generator/batch_normalization_2/ReadVariableOp_1
?Generator/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?Generator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp
AGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
AGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Г
0Generator/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_2/LeakyRelu:activations:06Generator/batch_normalization_2/ReadVariableOp:value:08Generator/batch_normalization_2/ReadVariableOp_1:value:0GGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџhX:::::*
epsilon%o:*
is_training( 22
0Generator/batch_normalization_2/FusedBatchNormV3
Generator/covtr6/ShapeShape4Generator/batch_normalization_2/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr6/Shape
$Generator/covtr6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr6/strided_slice/stack
&Generator/covtr6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr6/strided_slice/stack_1
&Generator/covtr6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr6/strided_slice/stack_2Ш
Generator/covtr6/strided_sliceStridedSliceGenerator/covtr6/Shape:output:0-Generator/covtr6/strided_slice/stack:output:0/Generator/covtr6/strided_slice/stack_1:output:0/Generator/covtr6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Generator/covtr6/strided_slicev
Generator/covtr6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :k2
Generator/covtr6/stack/1v
Generator/covtr6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Y2
Generator/covtr6/stack/2v
Generator/covtr6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Generator/covtr6/stack/3ј
Generator/covtr6/stackPack'Generator/covtr6/strided_slice:output:0!Generator/covtr6/stack/1:output:0!Generator/covtr6/stack/2:output:0!Generator/covtr6/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr6/stack
&Generator/covtr6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr6/strided_slice_1/stack
(Generator/covtr6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr6/strided_slice_1/stack_1
(Generator/covtr6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr6/strided_slice_1/stack_2в
 Generator/covtr6/strided_slice_1StridedSliceGenerator/covtr6/stack:output:0/Generator/covtr6/strided_slice_1/stack:output:01Generator/covtr6/strided_slice_1/stack_1:output:01Generator/covtr6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr6/strided_slice_1ц
0Generator/covtr6/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype022
0Generator/covtr6/conv2d_transpose/ReadVariableOpб
!Generator/covtr6/conv2d_transposeConv2DBackpropInputGenerator/covtr6/stack:output:08Generator/covtr6/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџkY@*
paddingVALID*
strides
2#
!Generator/covtr6/conv2d_transposeП
'Generator/covtr6/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'Generator/covtr6/BiasAdd/ReadVariableOpж
Generator/covtr6/BiasAddBiasAdd*Generator/covtr6/conv2d_transpose:output:0/Generator/covtr6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџkY@2
Generator/covtr6/BiasAddЇ
!Generator/leaky_re_lu_3/LeakyRelu	LeakyRelu!Generator/covtr6/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџkY@2#
!Generator/leaky_re_lu_3/LeakyReluд
.Generator/batch_normalization_3/ReadVariableOpReadVariableOp7generator_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype020
.Generator/batch_normalization_3/ReadVariableOpк
0Generator/batch_normalization_3/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0Generator/batch_normalization_3/ReadVariableOp_1
?Generator/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02A
?Generator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp
AGenerator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02C
AGenerator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Г
0Generator/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_3/LeakyRelu:activations:06Generator/batch_normalization_3/ReadVariableOp:value:08Generator/batch_normalization_3/ReadVariableOp_1:value:0GGenerator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџkY@:@:@:@:@:*
epsilon%o:*
is_training( 22
0Generator/batch_normalization_3/FusedBatchNormV3
Generator/covtr10/ShapeShape4Generator/batch_normalization_3/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr10/Shape
%Generator/covtr10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Generator/covtr10/strided_slice/stack
'Generator/covtr10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/covtr10/strided_slice/stack_1
'Generator/covtr10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/covtr10/strided_slice/stack_2Ю
Generator/covtr10/strided_sliceStridedSlice Generator/covtr10/Shape:output:0.Generator/covtr10/strided_slice/stack:output:00Generator/covtr10/strided_slice/stack_1:output:00Generator/covtr10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
Generator/covtr10/strided_slicex
Generator/covtr10/stack/1Const*
_output_shapes
: *
dtype0*
value	B :m2
Generator/covtr10/stack/1x
Generator/covtr10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Y2
Generator/covtr10/stack/2x
Generator/covtr10/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr10/stack/3ў
Generator/covtr10/stackPack(Generator/covtr10/strided_slice:output:0"Generator/covtr10/stack/1:output:0"Generator/covtr10/stack/2:output:0"Generator/covtr10/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr10/stack
'Generator/covtr10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'Generator/covtr10/strided_slice_1/stack 
)Generator/covtr10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)Generator/covtr10/strided_slice_1/stack_1 
)Generator/covtr10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)Generator/covtr10/strided_slice_1/stack_2и
!Generator/covtr10/strided_slice_1StridedSlice Generator/covtr10/stack:output:00Generator/covtr10/strided_slice_1/stack:output:02Generator/covtr10/strided_slice_1/stack_1:output:02Generator/covtr10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!Generator/covtr10/strided_slice_1щ
1Generator/covtr10/conv2d_transpose/ReadVariableOpReadVariableOp:generator_covtr10_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype023
1Generator/covtr10/conv2d_transpose/ReadVariableOpе
"Generator/covtr10/conv2d_transposeConv2DBackpropInput Generator/covtr10/stack:output:09Generator/covtr10/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџmY*
paddingVALID*
strides
2$
"Generator/covtr10/conv2d_transposeТ
(Generator/covtr10/BiasAdd/ReadVariableOpReadVariableOp1generator_covtr10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Generator/covtr10/BiasAdd/ReadVariableOpк
Generator/covtr10/BiasAddBiasAdd+Generator/covtr10/conv2d_transpose:output:00Generator/covtr10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџmY2
Generator/covtr10/BiasAdd
Generator/covtr10/SigmoidSigmoid"Generator/covtr10/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџmY2
Generator/covtr10/Sigmoidy
IdentityIdentityGenerator/covtr10/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџmY2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd:::::::::::::::::::::::::::::R N
'
_output_shapes
:џџџџџџџџџd
#
_user_specified_name	gen_noise
У
{
&__inference_covtr3_layer_call_fn_90506

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr3_layer_call_and_return_conditional_losses_904962
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
л
z
%__inference_Dense_layer_call_fn_92293

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ­*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_Dense_layer_call_and_return_conditional_losses_911222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ­2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџd::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ђ
Ј
5__inference_batch_normalization_3_layer_call_fn_92608

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_910472
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
х
^
B__inference_reshape_layer_call_and_return_conditional_losses_91152

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
Reshape/shape/3К
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ­:P L
(
_output_shapes
:џџџџџџџџџ­
 
_user_specified_nameinputs
С$
Г
A__inference_covtr6_layer_call_and_return_conditional_losses_90944

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityD
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
value	B :2
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
value	B :@2	
stack/3
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
strided_slice_3Г
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_transpose/ReadVariableOpё
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingVALID*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЄ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ц
Ћ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_90568

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%ЭЬL>2
FusedBatchNormV3џ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ј
b
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_92317

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
 
Ј
5__inference_batch_normalization_2_layer_call_fn_92521

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_908642
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ђ!
Г
A__inference_covtr3_layer_call_and_return_conditional_losses_90496

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityD
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
value	B :2	
stack/3
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
strided_slice_3Г
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp№
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЄ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
x
ф
!__inference__traced_restore_92809
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias$
 assignvariableop_2_covtr3_kernel"
assignvariableop_3_covtr3_bias0
,assignvariableop_4_batch_normalization_gamma/
+assignvariableop_5_batch_normalization_beta6
2assignvariableop_6_batch_normalization_moving_mean:
6assignvariableop_7_batch_normalization_moving_variance$
 assignvariableop_8_covtr4_kernel"
assignvariableop_9_covtr4_bias3
/assignvariableop_10_batch_normalization_1_gamma2
.assignvariableop_11_batch_normalization_1_beta9
5assignvariableop_12_batch_normalization_1_moving_mean=
9assignvariableop_13_batch_normalization_1_moving_variance%
!assignvariableop_14_covtr5_kernel#
assignvariableop_15_covtr5_bias3
/assignvariableop_16_batch_normalization_2_gamma2
.assignvariableop_17_batch_normalization_2_beta9
5assignvariableop_18_batch_normalization_2_moving_mean=
9assignvariableop_19_batch_normalization_2_moving_variance%
!assignvariableop_20_covtr6_kernel#
assignvariableop_21_covtr6_bias3
/assignvariableop_22_batch_normalization_3_gamma2
.assignvariableop_23_batch_normalization_3_beta9
5assignvariableop_24_batch_normalization_3_moving_mean=
9assignvariableop_25_batch_normalization_3_moving_variance&
"assignvariableop_26_covtr10_kernel$
 assignvariableop_27_covtr10_bias
identity_29ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9ч
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ѓ
valueщBцB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesШ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesН
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ђ
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ѕ
AssignVariableOp_2AssignVariableOp assignvariableop_2_covtr3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѓ
AssignVariableOp_3AssignVariableOpassignvariableop_3_covtr3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Б
AssignVariableOp_4AssignVariableOp,assignvariableop_4_batch_normalization_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5А
AssignVariableOp_5AssignVariableOp+assignvariableop_5_batch_normalization_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6З
AssignVariableOp_6AssignVariableOp2assignvariableop_6_batch_normalization_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Л
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ѕ
AssignVariableOp_8AssignVariableOp assignvariableop_8_covtr4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ѓ
AssignVariableOp_9AssignVariableOpassignvariableop_9_covtr4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10З
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_1_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ж
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_1_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Н
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_1_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13С
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_1_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Љ
AssignVariableOp_14AssignVariableOp!assignvariableop_14_covtr5_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ї
AssignVariableOp_15AssignVariableOpassignvariableop_15_covtr5_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16З
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_2_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ж
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_2_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Н
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_2_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19С
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_2_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Љ
AssignVariableOp_20AssignVariableOp!assignvariableop_20_covtr6_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ї
AssignVariableOp_21AssignVariableOpassignvariableop_21_covtr6_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22З
AssignVariableOp_22AssignVariableOp/assignvariableop_22_batch_normalization_3_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ж
AssignVariableOp_23AssignVariableOp.assignvariableop_23_batch_normalization_3_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Н
AssignVariableOp_24AssignVariableOp5assignvariableop_24_batch_normalization_3_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25С
AssignVariableOp_25AssignVariableOp9assignvariableop_25_batch_normalization_3_moving_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Њ
AssignVariableOp_26AssignVariableOp"assignvariableop_26_covtr10_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ј
AssignVariableOp_27AssignVariableOp assignvariableop_27_covtr10_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_279
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЦ
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_28Й
Identity_29IdentityIdentity_28:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_29"#
identity_29Identity_29:output:0*
_input_shapest
r: ::::::::::::::::::::::::::::2$
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
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272(
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
Ј
b
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_91170

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

G
+__inference_leaky_re_lu_layer_call_fn_92322

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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_911702
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


N__inference_batch_normalization_layer_call_and_return_conditional_losses_92360

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ц
Ћ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_92342

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%ЭЬL>2
FusedBatchNormV3џ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ш
­
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_90716

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%ЭЬL>2
FusedBatchNormV3џ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ѓ

)__inference_Generator_layer_call_fn_92212

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

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identityЂStatefulPartitionedCallю
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_Generator_layer_call_and_return_conditional_losses_915312
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Њ
d
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_91329

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs


P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_90747

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
У
{
&__inference_covtr5_layer_call_fn_90802

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr5_layer_call_and_return_conditional_losses_907922
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ж

#__inference_signature_wrapper_91789
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

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identityЂStatefulPartitionedCallУ
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџmY*>
_read_only_resource_inputs 
	
*2
config_proto" 

CPU

GPU2 *0J 8 *)
f$R"
 __inference__wrapped_model_904622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџmY2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:џџџџџџџџџd
#
_user_specified_name	gen_noise
т

D__inference_Generator_layer_call_and_return_conditional_losses_91974

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource3
/covtr3_conv2d_transpose_readvariableop_resource*
&covtr3_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource3
/covtr4_conv2d_transpose_readvariableop_resource*
&covtr4_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource3
/covtr5_conv2d_transpose_readvariableop_resource*
&covtr5_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource3
/covtr6_conv2d_transpose_readvariableop_resource*
&covtr6_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource4
0covtr10_conv2d_transpose_readvariableop_resource+
'covtr10_biasadd_readvariableop_resource
identityЂ"batch_normalization/AssignNewValueЂ$batch_normalization/AssignNewValue_1Ђ$batch_normalization_1/AssignNewValueЂ&batch_normalization_1/AssignNewValue_1Ђ$batch_normalization_2/AssignNewValueЂ&batch_normalization_2/AssignNewValue_1Ђ$batch_normalization_3/AssignNewValueЂ&batch_normalization_3/AssignNewValue_1 
Dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	d­*
dtype02
Dense/MatMul/ReadVariableOp
Dense/MatMulMatMulinputs#Dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ­2
Dense/MatMul
Dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:­*
dtype02
Dense/BiasAdd/ReadVariableOp
Dense/BiasAddBiasAddDense/MatMul:product:0$Dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ­2
Dense/BiasAddk

Dense/ReluReluDense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ­2

Dense/Reluf
reshape/ShapeShapeDense/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
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
reshape/Reshape/shapeЁ
reshape/ReshapeReshapeDense/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
reshape/Reshaped
covtr3/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
covtr3/Shape
covtr3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr3/strided_slice/stack
covtr3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr3/strided_slice/stack_1
covtr3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr3/strided_slice/stack_2
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
value	B :2
covtr3/stack/1b
covtr3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
covtr3/stack/2b
covtr3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr3/stack/3М
covtr3/stackPackcovtr3/strided_slice:output:0covtr3/stack/1:output:0covtr3/stack/2:output:0covtr3/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr3/stack
covtr3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr3/strided_slice_1/stack
covtr3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr3/strided_slice_1/stack_1
covtr3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr3/strided_slice_1/stack_2
covtr3/strided_slice_1StridedSlicecovtr3/stack:output:0%covtr3/strided_slice_1/stack:output:0'covtr3/strided_slice_1/stack_1:output:0'covtr3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr3/strided_slice_1Ш
&covtr3/conv2d_transpose/ReadVariableOpReadVariableOp/covtr3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr3/conv2d_transpose/ReadVariableOp
covtr3/conv2d_transposeConv2DBackpropInputcovtr3/stack:output:0.covtr3/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
covtr3/conv2d_transposeЁ
covtr3/BiasAdd/ReadVariableOpReadVariableOp&covtr3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr3/BiasAdd/ReadVariableOpЎ
covtr3/BiasAddBiasAdd covtr3/conv2d_transpose:output:0%covtr3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
covtr3/BiasAdd
leaky_re_lu/LeakyRelu	LeakyRelucovtr3/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ2
leaky_re_lu/LeakyReluА
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOpЖ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1э
$batch_normalization/FusedBatchNormV3FusedBatchNormV3#leaky_re_lu/LeakyRelu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%ЭЬL>2&
$batch_normalization/FusedBatchNormV3ї
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1t
covtr4/ShapeShape(batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr4/Shape
covtr4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr4/strided_slice/stack
covtr4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr4/strided_slice/stack_1
covtr4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr4/strided_slice/stack_2
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
value	B :42
covtr4/stack/1b
covtr4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :,2
covtr4/stack/2b
covtr4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr4/stack/3М
covtr4/stackPackcovtr4/strided_slice:output:0covtr4/stack/1:output:0covtr4/stack/2:output:0covtr4/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr4/stack
covtr4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr4/strided_slice_1/stack
covtr4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr4/strided_slice_1/stack_1
covtr4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr4/strided_slice_1/stack_2
covtr4/strided_slice_1StridedSlicecovtr4/stack:output:0%covtr4/strided_slice_1/stack:output:0'covtr4/strided_slice_1/stack_1:output:0'covtr4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr4/strided_slice_1Ш
&covtr4/conv2d_transpose/ReadVariableOpReadVariableOp/covtr4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr4/conv2d_transpose/ReadVariableOp
covtr4/conv2d_transposeConv2DBackpropInputcovtr4/stack:output:0.covtr4/conv2d_transpose/ReadVariableOp:value:0(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ4,*
paddingSAME*
strides
2
covtr4/conv2d_transposeЁ
covtr4/BiasAdd/ReadVariableOpReadVariableOp&covtr4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr4/BiasAdd/ReadVariableOpЎ
covtr4/BiasAddBiasAdd covtr4/conv2d_transpose:output:0%covtr4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ4,2
covtr4/BiasAdd
leaky_re_lu_1/LeakyRelu	LeakyRelucovtr4/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ4,2
leaky_re_lu_1/LeakyReluЖ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOpМ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ћ
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_1/LeakyRelu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ4,:::::*
epsilon%o:*
exponential_avg_factor%ЭЬL>2(
&batch_normalization_1/FusedBatchNormV3
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1v
covtr5/ShapeShape*batch_normalization_1/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr5/Shape
covtr5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr5/strided_slice/stack
covtr5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr5/strided_slice/stack_1
covtr5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr5/strided_slice/stack_2
covtr5/strided_sliceStridedSlicecovtr5/Shape:output:0#covtr5/strided_slice/stack:output:0%covtr5/strided_slice/stack_1:output:0%covtr5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr5/strided_sliceb
covtr5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :h2
covtr5/stack/1b
covtr5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :X2
covtr5/stack/2b
covtr5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr5/stack/3М
covtr5/stackPackcovtr5/strided_slice:output:0covtr5/stack/1:output:0covtr5/stack/2:output:0covtr5/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr5/stack
covtr5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr5/strided_slice_1/stack
covtr5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr5/strided_slice_1/stack_1
covtr5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr5/strided_slice_1/stack_2
covtr5/strided_slice_1StridedSlicecovtr5/stack:output:0%covtr5/strided_slice_1/stack:output:0'covtr5/strided_slice_1/stack_1:output:0'covtr5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr5/strided_slice_1Ш
&covtr5/conv2d_transpose/ReadVariableOpReadVariableOp/covtr5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr5/conv2d_transpose/ReadVariableOp
covtr5/conv2d_transposeConv2DBackpropInputcovtr5/stack:output:0.covtr5/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџhX*
paddingSAME*
strides
2
covtr5/conv2d_transposeЁ
covtr5/BiasAdd/ReadVariableOpReadVariableOp&covtr5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr5/BiasAdd/ReadVariableOpЎ
covtr5/BiasAddBiasAdd covtr5/conv2d_transpose:output:0%covtr5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhX2
covtr5/BiasAdd
leaky_re_lu_2/LeakyRelu	LeakyRelucovtr5/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџhX2
leaky_re_lu_2/LeakyReluЖ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_2/ReadVariableOpМ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1щ
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ћ
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_2/LeakyRelu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџhX:::::*
epsilon%o:*
exponential_avg_factor%ЭЬL>2(
&batch_normalization_2/FusedBatchNormV3
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1v
covtr6/ShapeShape*batch_normalization_2/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr6/Shape
covtr6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr6/strided_slice/stack
covtr6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr6/strided_slice/stack_1
covtr6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr6/strided_slice/stack_2
covtr6/strided_sliceStridedSlicecovtr6/Shape:output:0#covtr6/strided_slice/stack:output:0%covtr6/strided_slice/stack_1:output:0%covtr6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr6/strided_sliceb
covtr6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :k2
covtr6/stack/1b
covtr6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Y2
covtr6/stack/2b
covtr6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
covtr6/stack/3М
covtr6/stackPackcovtr6/strided_slice:output:0covtr6/stack/1:output:0covtr6/stack/2:output:0covtr6/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr6/stack
covtr6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr6/strided_slice_1/stack
covtr6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr6/strided_slice_1/stack_1
covtr6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr6/strided_slice_1/stack_2
covtr6/strided_slice_1StridedSlicecovtr6/stack:output:0%covtr6/strided_slice_1/stack:output:0'covtr6/strided_slice_1/stack_1:output:0'covtr6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr6/strided_slice_1Ш
&covtr6/conv2d_transpose/ReadVariableOpReadVariableOp/covtr6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&covtr6/conv2d_transpose/ReadVariableOp
covtr6/conv2d_transposeConv2DBackpropInputcovtr6/stack:output:0.covtr6/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџkY@*
paddingVALID*
strides
2
covtr6/conv2d_transposeЁ
covtr6/BiasAdd/ReadVariableOpReadVariableOp&covtr6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
covtr6/BiasAdd/ReadVariableOpЎ
covtr6/BiasAddBiasAdd covtr6/conv2d_transpose:output:0%covtr6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџkY@2
covtr6/BiasAdd
leaky_re_lu_3/LeakyRelu	LeakyRelucovtr6/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџkY@2
leaky_re_lu_3/LeakyReluЖ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOpМ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1щ
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ћ
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_3/LeakyRelu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџkY@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%ЭЬL>2(
&batch_normalization_3/FusedBatchNormV3
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1x
covtr10/ShapeShape*batch_normalization_3/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr10/Shape
covtr10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr10/strided_slice/stack
covtr10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr10/strided_slice/stack_1
covtr10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr10/strided_slice/stack_2
covtr10/strided_sliceStridedSlicecovtr10/Shape:output:0$covtr10/strided_slice/stack:output:0&covtr10/strided_slice/stack_1:output:0&covtr10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr10/strided_sliced
covtr10/stack/1Const*
_output_shapes
: *
dtype0*
value	B :m2
covtr10/stack/1d
covtr10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Y2
covtr10/stack/2d
covtr10/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr10/stack/3Т
covtr10/stackPackcovtr10/strided_slice:output:0covtr10/stack/1:output:0covtr10/stack/2:output:0covtr10/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr10/stack
covtr10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr10/strided_slice_1/stack
covtr10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr10/strided_slice_1/stack_1
covtr10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr10/strided_slice_1/stack_2
covtr10/strided_slice_1StridedSlicecovtr10/stack:output:0&covtr10/strided_slice_1/stack:output:0(covtr10/strided_slice_1/stack_1:output:0(covtr10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr10/strided_slice_1Ы
'covtr10/conv2d_transpose/ReadVariableOpReadVariableOp0covtr10_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02)
'covtr10/conv2d_transpose/ReadVariableOpЃ
covtr10/conv2d_transposeConv2DBackpropInputcovtr10/stack:output:0/covtr10/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџmY*
paddingVALID*
strides
2
covtr10/conv2d_transposeЄ
covtr10/BiasAdd/ReadVariableOpReadVariableOp'covtr10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
covtr10/BiasAdd/ReadVariableOpВ
covtr10/BiasAddBiasAdd!covtr10/conv2d_transpose:output:0&covtr10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџmY2
covtr10/BiasAdd
covtr10/SigmoidSigmoidcovtr10/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџmY2
covtr10/SigmoidЋ
IdentityIdentitycovtr10/Sigmoid:y:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_1*
T0*/
_output_shapes
:џџџџџџџџџmY2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd::::::::::::::::::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_1:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs


P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_91047

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
эM
Ч	
D__inference_Generator_layer_call_and_return_conditional_losses_91531

inputs
dense_91459
dense_91461
covtr3_91465
covtr3_91467
batch_normalization_91471
batch_normalization_91473
batch_normalization_91475
batch_normalization_91477
covtr4_91480
covtr4_91482
batch_normalization_1_91486
batch_normalization_1_91488
batch_normalization_1_91490
batch_normalization_1_91492
covtr5_91495
covtr5_91497
batch_normalization_2_91501
batch_normalization_2_91503
batch_normalization_2_91505
batch_normalization_2_91507
covtr6_91510
covtr6_91512
batch_normalization_3_91516
batch_normalization_3_91518
batch_normalization_3_91520
batch_normalization_3_91522
covtr10_91525
covtr10_91527
identityЂDense/StatefulPartitionedCallЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂcovtr10/StatefulPartitionedCallЂcovtr3/StatefulPartitionedCallЂcovtr4/StatefulPartitionedCallЂcovtr5/StatefulPartitionedCallЂcovtr6/StatefulPartitionedCall
Dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_91459dense_91461*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ­*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_Dense_layer_call_and_return_conditional_losses_911222
Dense/StatefulPartitionedCall§
reshape/PartitionedCallPartitionedCall&Dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_911522
reshape/PartitionedCallР
covtr3/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr3_91465covtr3_91467*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr3_layer_call_and_return_conditional_losses_904962 
covtr3/StatefulPartitionedCall
leaky_re_lu/PartitionedCallPartitionedCall'covtr3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_911702
leaky_re_lu/PartitionedCallН
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_91471batch_normalization_91473batch_normalization_91475batch_normalization_91477*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_905682-
+batch_normalization/StatefulPartitionedCallд
covtr4/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr4_91480covtr4_91482*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr4_layer_call_and_return_conditional_losses_906442 
covtr4/StatefulPartitionedCallЂ
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_912232
leaky_re_lu_1/PartitionedCallЭ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_91486batch_normalization_1_91488batch_normalization_1_91490batch_normalization_1_91492*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_907162/
-batch_normalization_1/StatefulPartitionedCallж
covtr5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr5_91495covtr5_91497*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr5_layer_call_and_return_conditional_losses_907922 
covtr5/StatefulPartitionedCallЂ
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_912762
leaky_re_lu_2/PartitionedCallЭ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_91501batch_normalization_2_91503batch_normalization_2_91505batch_normalization_2_91507*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_908642/
-batch_normalization_2/StatefulPartitionedCallж
covtr6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0covtr6_91510covtr6_91512*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr6_layer_call_and_return_conditional_losses_909442 
covtr6/StatefulPartitionedCallЂ
leaky_re_lu_3/PartitionedCallPartitionedCall'covtr6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_913292
leaky_re_lu_3/PartitionedCallЭ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0batch_normalization_3_91516batch_normalization_3_91518batch_normalization_3_91520batch_normalization_3_91522*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_910162/
-batch_normalization_3/StatefulPartitionedCallл
covtr10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0covtr10_91525covtr10_91527*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_covtr10_layer_call_and_return_conditional_losses_910972!
covtr10/StatefulPartitionedCall
IdentityIdentity(covtr10/StatefulPartitionedCall:output:0^Dense/StatefulPartitionedCall,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^covtr10/StatefulPartitionedCall^covtr3/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^covtr5/StatefulPartitionedCall^covtr6/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd::::::::::::::::::::::::::::2>
Dense/StatefulPartitionedCallDense/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
covtr10/StatefulPartitionedCallcovtr10/StatefulPartitionedCall2@
covtr3/StatefulPartitionedCallcovtr3/StatefulPartitionedCall2@
covtr4/StatefulPartitionedCallcovtr4/StatefulPartitionedCall2@
covtr5/StatefulPartitionedCallcovtr5/StatefulPartitionedCall2@
covtr6/StatefulPartitionedCallcovtr6/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
К%
Д
B__inference_covtr10_layer_call_and_return_conditional_losses_91097

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityD
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
value	B :2
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
value	B : 2	
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
value	B :2	
stack/3
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
strided_slice_3Г
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_transpose/ReadVariableOpё
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingVALID*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЄ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
Sigmoidy
IdentityIdentitySigmoid:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
х
^
B__inference_reshape_layer_call_and_return_conditional_losses_92307

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
Reshape/shape/3К
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ­:P L
(
_output_shapes
:џџџџџџџџџ­
 
_user_specified_nameinputs
У
{
&__inference_covtr4_layer_call_fn_90654

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr4_layer_call_and_return_conditional_losses_906442
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ђ
Ј
5__inference_batch_normalization_2_layer_call_fn_92534

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_908952
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ќ

)__inference_Generator_layer_call_fn_91590
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

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identityЂStatefulPartitionedCallё
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_Generator_layer_call_and_return_conditional_losses_915312
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:џџџџџџџџџd
#
_user_specified_name	gen_noise

І
3__inference_batch_normalization_layer_call_fn_92386

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_905992
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
іM
Ъ	
D__inference_Generator_layer_call_and_return_conditional_losses_91378
	gen_noise
dense_91133
dense_91135
covtr3_91160
covtr3_91162
batch_normalization_91204
batch_normalization_91206
batch_normalization_91208
batch_normalization_91210
covtr4_91213
covtr4_91215
batch_normalization_1_91257
batch_normalization_1_91259
batch_normalization_1_91261
batch_normalization_1_91263
covtr5_91266
covtr5_91268
batch_normalization_2_91310
batch_normalization_2_91312
batch_normalization_2_91314
batch_normalization_2_91316
covtr6_91319
covtr6_91321
batch_normalization_3_91363
batch_normalization_3_91365
batch_normalization_3_91367
batch_normalization_3_91369
covtr10_91372
covtr10_91374
identityЂDense/StatefulPartitionedCallЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂcovtr10/StatefulPartitionedCallЂcovtr3/StatefulPartitionedCallЂcovtr4/StatefulPartitionedCallЂcovtr5/StatefulPartitionedCallЂcovtr6/StatefulPartitionedCall
Dense/StatefulPartitionedCallStatefulPartitionedCall	gen_noisedense_91133dense_91135*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ­*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_Dense_layer_call_and_return_conditional_losses_911222
Dense/StatefulPartitionedCall§
reshape/PartitionedCallPartitionedCall&Dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_911522
reshape/PartitionedCallР
covtr3/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr3_91160covtr3_91162*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr3_layer_call_and_return_conditional_losses_904962 
covtr3/StatefulPartitionedCall
leaky_re_lu/PartitionedCallPartitionedCall'covtr3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_911702
leaky_re_lu/PartitionedCallН
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_91204batch_normalization_91206batch_normalization_91208batch_normalization_91210*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_905682-
+batch_normalization/StatefulPartitionedCallд
covtr4/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr4_91213covtr4_91215*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr4_layer_call_and_return_conditional_losses_906442 
covtr4/StatefulPartitionedCallЂ
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_912232
leaky_re_lu_1/PartitionedCallЭ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_91257batch_normalization_1_91259batch_normalization_1_91261batch_normalization_1_91263*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_907162/
-batch_normalization_1/StatefulPartitionedCallж
covtr5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr5_91266covtr5_91268*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr5_layer_call_and_return_conditional_losses_907922 
covtr5/StatefulPartitionedCallЂ
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_912762
leaky_re_lu_2/PartitionedCallЭ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_91310batch_normalization_2_91312batch_normalization_2_91314batch_normalization_2_91316*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_908642/
-batch_normalization_2/StatefulPartitionedCallж
covtr6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0covtr6_91319covtr6_91321*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_covtr6_layer_call_and_return_conditional_losses_909442 
covtr6/StatefulPartitionedCallЂ
leaky_re_lu_3/PartitionedCallPartitionedCall'covtr6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_913292
leaky_re_lu_3/PartitionedCallЭ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0batch_normalization_3_91363batch_normalization_3_91365batch_normalization_3_91367batch_normalization_3_91369*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_910162/
-batch_normalization_3/StatefulPartitionedCallл
covtr10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0covtr10_91372covtr10_91374*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_covtr10_layer_call_and_return_conditional_losses_910972!
covtr10/StatefulPartitionedCall
IdentityIdentity(covtr10/StatefulPartitionedCall:output:0^Dense/StatefulPartitionedCall,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^covtr10/StatefulPartitionedCall^covtr3/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^covtr5/StatefulPartitionedCall^covtr6/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd::::::::::::::::::::::::::::2>
Dense/StatefulPartitionedCallDense/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
covtr10/StatefulPartitionedCallcovtr10/StatefulPartitionedCall2@
covtr3/StatefulPartitionedCallcovtr3/StatefulPartitionedCall2@
covtr4/StatefulPartitionedCallcovtr4/StatefulPartitionedCall2@
covtr5/StatefulPartitionedCallcovtr5/StatefulPartitionedCall2@
covtr6/StatefulPartitionedCallcovtr6/StatefulPartitionedCall:R N
'
_output_shapes
:џџџџџџџџџd
#
_user_specified_name	gen_noise


P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_90895

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
 
Ј
5__inference_batch_normalization_3_layer_call_fn_92595

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_910162
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Њ
d
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_92465

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


)__inference_Generator_layer_call_fn_91726
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

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identityЂStatefulPartitionedCallљ
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*>
_read_only_resource_inputs 
	
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_Generator_layer_call_and_return_conditional_losses_916672
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:џџџџџџџџџd
#
_user_specified_name	gen_noise"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ж
serving_defaultЂ
?
	gen_noise2
serving_default_gen_noise:0џџџџџџџџџdC
covtr108
StatefulPartitionedCall:0џџџџџџџџџmYtensorflow/serving/predict:џђ
ж
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
trainable_variables
regularization_losses
	variables
	keras_api

signatures
Т_default_save_signature
У__call__
+Ф&call_and_return_all_conditional_losses"М
_tf_keras_network{"class_name": "Functional", "name": "Generator", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Generator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gen_noise"}, "name": "gen_noise", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "Dense", "trainable": true, "dtype": "float32", "units": 429, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense", "inbound_nodes": [[["gen_noise", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [13, 11, 3]}}, "name": "reshape", "inbound_nodes": [[["Dense", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr3", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr3", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu", "inbound_nodes": [[["covtr3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr4", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_1", "inbound_nodes": [[["covtr4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr5", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr5", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_2", "inbound_nodes": [[["covtr5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr6", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_3", "inbound_nodes": [[["covtr6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["leaky_re_lu_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr10", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr10", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}], "input_layers": [["gen_noise", 0, 0]], "output_layers": [["covtr10", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Generator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gen_noise"}, "name": "gen_noise", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "Dense", "trainable": true, "dtype": "float32", "units": 429, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense", "inbound_nodes": [[["gen_noise", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [13, 11, 3]}}, "name": "reshape", "inbound_nodes": [[["Dense", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr3", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr3", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu", "inbound_nodes": [[["covtr3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr4", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_1", "inbound_nodes": [[["covtr4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr5", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr5", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_2", "inbound_nodes": [[["covtr5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr6", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_3", "inbound_nodes": [[["covtr6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["leaky_re_lu_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr10", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr10", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}], "input_layers": [["gen_noise", 0, 0]], "output_layers": [["covtr10", 0, 0]]}}}
ё"ю
_tf_keras_input_layerЮ{"class_name": "InputLayer", "name": "gen_noise", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gen_noise"}}
ё

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"Ъ
_tf_keras_layerА{"class_name": "Dense", "name": "Dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dense", "trainable": true, "dtype": "float32", "units": 429, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
ї
trainable_variables
regularization_losses
	variables
	keras_api
+Ч&call_and_return_all_conditional_losses
Ш__call__"ц
_tf_keras_layerЬ{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [13, 11, 3]}}}
Ќ


 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
+Щ&call_and_return_all_conditional_losses
Ъ__call__"	
_tf_keras_layerы{"class_name": "Conv2DTranspose", "name": "covtr3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr3", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 11, 3]}}
м
&trainable_variables
'regularization_losses
(	variables
)	keras_api
+Ы&call_and_return_all_conditional_losses
Ь__call__"Ы
_tf_keras_layerБ{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Е	
*axis
	+gamma
,beta
-moving_mean
.moving_variance
/trainable_variables
0regularization_losses
1	variables
2	keras_api
+Э&call_and_return_all_conditional_losses
Ю__call__"п
_tf_keras_layerХ{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 22, 8]}}
­


3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
+Я&call_and_return_all_conditional_losses
а__call__"	
_tf_keras_layerь{"class_name": "Conv2DTranspose", "name": "covtr4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 22, 8]}}
р
9trainable_variables
:regularization_losses
;	variables
<	keras_api
+б&call_and_return_all_conditional_losses
в__call__"Я
_tf_keras_layerЕ{"class_name": "LeakyReLU", "name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Л	
=axis
	>gamma
?beta
@moving_mean
Amoving_variance
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
+г&call_and_return_all_conditional_losses
д__call__"х
_tf_keras_layerЫ{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 52, 44, 16]}}
Џ


Fkernel
Gbias
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"	
_tf_keras_layerю{"class_name": "Conv2DTranspose", "name": "covtr5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr5", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 52, 44, 16]}}
р
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
+з&call_and_return_all_conditional_losses
и__call__"Я
_tf_keras_layerЕ{"class_name": "LeakyReLU", "name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
М	
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
+й&call_and_return_all_conditional_losses
к__call__"ц
_tf_keras_layerЬ{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 104, 88, 24]}}
Б


Ykernel
Zbias
[trainable_variables
\regularization_losses
]	variables
^	keras_api
+л&call_and_return_all_conditional_losses
м__call__"	
_tf_keras_layer№{"class_name": "Conv2DTranspose", "name": "covtr6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 104, 88, 24]}}
р
_trainable_variables
`regularization_losses
a	variables
b	keras_api
+н&call_and_return_all_conditional_losses
о__call__"Я
_tf_keras_layerЕ{"class_name": "LeakyReLU", "name": "leaky_re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
М	
caxis
	dgamma
ebeta
fmoving_mean
gmoving_variance
htrainable_variables
iregularization_losses
j	variables
k	keras_api
+п&call_and_return_all_conditional_losses
р__call__"ц
_tf_keras_layerЬ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 107, 89, 64]}}
Г


lkernel
mbias
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
+с&call_and_return_all_conditional_losses
т__call__"	
_tf_keras_layerђ{"class_name": "Conv2DTranspose", "name": "covtr10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr10", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 107, 89, 64]}}
Ж
0
1
 2
!3
+4
,5
36
47
>8
?9
F10
G11
Q12
R13
Y14
Z15
d16
e17
l18
m19"
trackable_list_wrapper
 "
trackable_list_wrapper
і
0
1
 2
!3
+4
,5
-6
.7
38
49
>10
?11
@12
A13
F14
G15
Q16
R17
S18
T19
Y20
Z21
d22
e23
f24
g25
l26
m27"
trackable_list_wrapper
Ю

rlayers
slayer_regularization_losses
tlayer_metrics
unon_trainable_variables
vmetrics
trainable_variables
regularization_losses
	variables
У__call__
Т_default_save_signature
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
-
уserving_default"
signature_map
:	d­2Dense/kernel
:­2
Dense/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А

wlayers
xlayer_regularization_losses
ylayer_metrics
znon_trainable_variables
{metrics
trainable_variables
regularization_losses
	variables
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Б

|layers
}layer_regularization_losses
~layer_metrics
non_trainable_variables
metrics
trainable_variables
regularization_losses
	variables
Ш__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
':%2covtr3/kernel
:2covtr3/bias
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
Е
layers
 layer_regularization_losses
layer_metrics
non_trainable_variables
metrics
"trainable_variables
#regularization_losses
$	variables
Ъ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layers
 layer_regularization_losses
layer_metrics
non_trainable_variables
metrics
&trainable_variables
'regularization_losses
(	variables
Ь__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
+0
,1
-2
.3"
trackable_list_wrapper
Е
layers
 layer_regularization_losses
layer_metrics
non_trainable_variables
metrics
/trainable_variables
0regularization_losses
1	variables
Ю__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
':%2covtr4/kernel
:2covtr4/bias
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
Е
layers
 layer_regularization_losses
layer_metrics
non_trainable_variables
metrics
5trainable_variables
6regularization_losses
7	variables
а__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layers
 layer_regularization_losses
layer_metrics
non_trainable_variables
metrics
9trainable_variables
:regularization_losses
;	variables
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_1/gamma
(:&2batch_normalization_1/beta
1:/ (2!batch_normalization_1/moving_mean
5:3 (2%batch_normalization_1/moving_variance
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
>0
?1
@2
A3"
trackable_list_wrapper
Е
layers
 layer_regularization_losses
layer_metrics
non_trainable_variables
metrics
Btrainable_variables
Cregularization_losses
D	variables
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
':%2covtr5/kernel
:2covtr5/bias
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
Е
layers
  layer_regularization_losses
Ёlayer_metrics
Ђnon_trainable_variables
Ѓmetrics
Htrainable_variables
Iregularization_losses
J	variables
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Єlayers
 Ѕlayer_regularization_losses
Іlayer_metrics
Їnon_trainable_variables
Јmetrics
Ltrainable_variables
Mregularization_losses
N	variables
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_2/gamma
(:&2batch_normalization_2/beta
1:/ (2!batch_normalization_2/moving_mean
5:3 (2%batch_normalization_2/moving_variance
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
Q0
R1
S2
T3"
trackable_list_wrapper
Е
Љlayers
 Њlayer_regularization_losses
Ћlayer_metrics
Ќnon_trainable_variables
­metrics
Utrainable_variables
Vregularization_losses
W	variables
к__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
':%@2covtr6/kernel
:@2covtr6/bias
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
Е
Ўlayers
 Џlayer_regularization_losses
Аlayer_metrics
Бnon_trainable_variables
Вmetrics
[trainable_variables
\regularization_losses
]	variables
м__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Гlayers
 Дlayer_regularization_losses
Еlayer_metrics
Жnon_trainable_variables
Зmetrics
_trainable_variables
`regularization_losses
a	variables
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_3/gamma
(:&@2batch_normalization_3/beta
1:/@ (2!batch_normalization_3/moving_mean
5:3@ (2%batch_normalization_3/moving_variance
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
d0
e1
f2
g3"
trackable_list_wrapper
Е
Иlayers
 Йlayer_regularization_losses
Кlayer_metrics
Лnon_trainable_variables
Мmetrics
htrainable_variables
iregularization_losses
j	variables
р__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
(:&@2covtr10/kernel
:2covtr10/bias
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
Е
Нlayers
 Оlayer_regularization_losses
Пlayer_metrics
Рnon_trainable_variables
Сmetrics
ntrainable_variables
oregularization_losses
p	variables
т__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object

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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
X
-0
.1
@2
A3
S4
T5
f6
g7"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
-0
.1"
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
.
@0
A1"
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
.
S0
T1"
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
.
f0
g1"
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
р2н
 __inference__wrapped_model_90462И
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *(Ђ%
# 
	gen_noiseџџџџџџџџџd
ђ2я
)__inference_Generator_layer_call_fn_92273
)__inference_Generator_layer_call_fn_92212
)__inference_Generator_layer_call_fn_91726
)__inference_Generator_layer_call_fn_91590Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
о2л
D__inference_Generator_layer_call_and_return_conditional_losses_91453
D__inference_Generator_layer_call_and_return_conditional_losses_91378
D__inference_Generator_layer_call_and_return_conditional_losses_91974
D__inference_Generator_layer_call_and_return_conditional_losses_92151Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ъ2ч
@__inference_Dense_layer_call_and_return_conditional_losses_92284Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Я2Ь
%__inference_Dense_layer_call_fn_92293Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_reshape_layer_call_and_return_conditional_losses_92307Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_reshape_layer_call_fn_92312Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 2
A__inference_covtr3_layer_call_and_return_conditional_losses_90496з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
&__inference_covtr3_layer_call_fn_90506з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
№2э
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_92317Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_leaky_re_lu_layer_call_fn_92322Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
к2з
N__inference_batch_normalization_layer_call_and_return_conditional_losses_92360
N__inference_batch_normalization_layer_call_and_return_conditional_losses_92342Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Є2Ё
3__inference_batch_normalization_layer_call_fn_92386
3__inference_batch_normalization_layer_call_fn_92373Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 2
A__inference_covtr4_layer_call_and_return_conditional_losses_90644з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
&__inference_covtr4_layer_call_fn_90654з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
ђ2я
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_92391Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_leaky_re_lu_1_layer_call_fn_92396Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
о2л
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_92434
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_92416Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2Ѕ
5__inference_batch_normalization_1_layer_call_fn_92447
5__inference_batch_normalization_1_layer_call_fn_92460Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 2
A__inference_covtr5_layer_call_and_return_conditional_losses_90792з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
&__inference_covtr5_layer_call_fn_90802з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
ђ2я
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_92465Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_leaky_re_lu_2_layer_call_fn_92470Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
о2л
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_92490
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_92508Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2Ѕ
5__inference_batch_normalization_2_layer_call_fn_92521
5__inference_batch_normalization_2_layer_call_fn_92534Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 2
A__inference_covtr6_layer_call_and_return_conditional_losses_90944з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
&__inference_covtr6_layer_call_fn_90954з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
ђ2я
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_92539Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_leaky_re_lu_3_layer_call_fn_92544Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
о2л
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_92564
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_92582Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2Ѕ
5__inference_batch_normalization_3_layer_call_fn_92595
5__inference_batch_normalization_3_layer_call_fn_92608Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ё2
B__inference_covtr10_layer_call_and_return_conditional_losses_91097з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
2
'__inference_covtr10_layer_call_fn_91107з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
4B2
#__inference_signature_wrapper_91789	gen_noiseЁ
@__inference_Dense_layer_call_and_return_conditional_losses_92284]/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "&Ђ#

0џџџџџџџџџ­
 y
%__inference_Dense_layer_call_fn_92293P/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџ­ф
D__inference_Generator_layer_call_and_return_conditional_losses_91378 !+,-.34>?@AFGQRSTYZdefglm:Ђ7
0Ђ-
# 
	gen_noiseџџџџџџџџџd
p

 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ф
D__inference_Generator_layer_call_and_return_conditional_losses_91453 !+,-.34>?@AFGQRSTYZdefglm:Ђ7
0Ђ-
# 
	gen_noiseџџџџџџџџџd
p 

 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Я
D__inference_Generator_layer_call_and_return_conditional_losses_91974 !+,-.34>?@AFGQRSTYZdefglm7Ђ4
-Ђ*
 
inputsџџџџџџџџџd
p

 
Њ "-Ђ*
# 
0џџџџџџџџџmY
 Я
D__inference_Generator_layer_call_and_return_conditional_losses_92151 !+,-.34>?@AFGQRSTYZdefglm7Ђ4
-Ђ*
 
inputsџџџџџџџџџd
p 

 
Њ "-Ђ*
# 
0џџџџџџџџџmY
 М
)__inference_Generator_layer_call_fn_91590 !+,-.34>?@AFGQRSTYZdefglm:Ђ7
0Ђ-
# 
	gen_noiseџџџџџџџџџd
p

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџМ
)__inference_Generator_layer_call_fn_91726 !+,-.34>?@AFGQRSTYZdefglm:Ђ7
0Ђ-
# 
	gen_noiseџџџџџџџџџd
p 

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЙ
)__inference_Generator_layer_call_fn_92212 !+,-.34>?@AFGQRSTYZdefglm7Ђ4
-Ђ*
 
inputsџџџџџџџџџd
p

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЙ
)__inference_Generator_layer_call_fn_92273 !+,-.34>?@AFGQRSTYZdefglm7Ђ4
-Ђ*
 
inputsџџџџџџџџџd
p 

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџВ
 __inference__wrapped_model_90462 !+,-.34>?@AFGQRSTYZdefglm2Ђ/
(Ђ%
# 
	gen_noiseџџџџџџџџџd
Њ "9Њ6
4
covtr10)&
covtr10џџџџџџџџџmYы
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_92416>?@AMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ы
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_92434>?@AMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 У
5__inference_batch_normalization_1_layer_call_fn_92447>?@AMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџУ
5__inference_batch_normalization_1_layer_call_fn_92460>?@AMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџы
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_92490QRSTMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ы
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_92508QRSTMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 У
5__inference_batch_normalization_2_layer_call_fn_92521QRSTMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџУ
5__inference_batch_normalization_2_layer_call_fn_92534QRSTMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџы
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_92564defgMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 ы
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_92582defgMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 У
5__inference_batch_normalization_3_layer_call_fn_92595defgMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@У
5__inference_batch_normalization_3_layer_call_fn_92608defgMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@щ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_92342+,-.MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 щ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_92360+,-.MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 С
3__inference_batch_normalization_layer_call_fn_92373+,-.MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџС
3__inference_batch_normalization_layer_call_fn_92386+,-.MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџз
B__inference_covtr10_layer_call_and_return_conditional_losses_91097lmIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Џ
'__inference_covtr10_layer_call_fn_91107lmIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџж
A__inference_covtr3_layer_call_and_return_conditional_losses_90496 !IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ў
&__inference_covtr3_layer_call_fn_90506 !IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџж
A__inference_covtr4_layer_call_and_return_conditional_losses_9064434IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ў
&__inference_covtr4_layer_call_fn_9065434IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџж
A__inference_covtr5_layer_call_and_return_conditional_losses_90792FGIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ў
&__inference_covtr5_layer_call_fn_90802FGIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџж
A__inference_covtr6_layer_call_and_return_conditional_losses_90944YZIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Ў
&__inference_covtr6_layer_call_fn_90954YZIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@й
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_92391IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 А
-__inference_leaky_re_lu_1_layer_call_fn_92396IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџй
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_92465IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 А
-__inference_leaky_re_lu_2_layer_call_fn_92470IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџй
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_92539IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 А
-__inference_leaky_re_lu_3_layer_call_fn_92544IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@з
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_92317IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ў
+__inference_leaky_re_lu_layer_call_fn_92322IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЇ
B__inference_reshape_layer_call_and_return_conditional_losses_92307a0Ђ-
&Ђ#
!
inputsџџџџџџџџџ­
Њ "-Ђ*
# 
0џџџџџџџџџ
 
'__inference_reshape_layer_call_fn_92312T0Ђ-
&Ђ#
!
inputsџџџџџџџџџ­
Њ " џџџџџџџџџТ
#__inference_signature_wrapper_91789 !+,-.34>?@AFGQRSTYZdefglm?Ђ<
Ђ 
5Њ2
0
	gen_noise# 
	gen_noiseџџџџџџџџџd"9Њ6
4
covtr10)&
covtr10џџџџџџџџџmY