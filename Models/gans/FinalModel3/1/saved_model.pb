═┘
═Б
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
dtypetypeѕ
Й
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
executor_typestring ѕ
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.3.12v2.3.0-54-gfcc4b966f18єЙ
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	dГ*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	dГ*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Г*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:Г*
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
і
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namebatch_normalization/gamma
Ѓ
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
: *
dtype0
ѕ
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namebatch_normalization/beta
Ђ
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0
ќ
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean
Ј
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0
ъ
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance
Ќ
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
: *
dtype0
~
covtr3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2 *
shared_namecovtr3/kernel
w
!covtr3/kernel/Read/ReadVariableOpReadVariableOpcovtr3/kernel*&
_output_shapes
:2 *
dtype0
n
covtr3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namecovtr3/bias
g
covtr3/bias/Read/ReadVariableOpReadVariableOpcovtr3/bias*
_output_shapes
:2*
dtype0
ј
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*,
shared_namebatch_normalization_1/gamma
Є
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:2*
dtype0
ї
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*+
shared_namebatch_normalization_1/beta
Ё
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:2*
dtype0
џ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*2
shared_name#!batch_normalization_1/moving_mean
Њ
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:2*
dtype0
б
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*6
shared_name'%batch_normalization_1/moving_variance
Џ
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:2*
dtype0
~
covtr4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:P2*
shared_namecovtr4/kernel
w
!covtr4/kernel/Read/ReadVariableOpReadVariableOpcovtr4/kernel*&
_output_shapes
:P2*
dtype0
n
covtr4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_namecovtr4/bias
g
covtr4/bias/Read/ReadVariableOpReadVariableOpcovtr4/bias*
_output_shapes
:P*
dtype0
ј
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*,
shared_namebatch_normalization_2/gamma
Є
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:P*
dtype0
ї
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_namebatch_normalization_2/beta
Ё
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:P*
dtype0
џ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*2
shared_name#!batch_normalization_2/moving_mean
Њ
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:P*
dtype0
б
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*6
shared_name'%batch_normalization_2/moving_variance
Џ
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:P*
dtype0
z
cov3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_namecov3/kernel
s
cov3/kernel/Read/ReadVariableOpReadVariableOpcov3/kernel*&
_output_shapes
:P*
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
л9
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*І9
valueЂ9B■8 Bэ8
я
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
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
R
#	variables
$regularization_losses
%trainable_variables
&	keras_api
Ќ
'axis
	(gamma
)beta
*moving_mean
+moving_variance
,	variables
-regularization_losses
.trainable_variables
/	keras_api
h

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
R
6	variables
7regularization_losses
8trainable_variables
9	keras_api
Ќ
:axis
	;gamma
<beta
=moving_mean
>moving_variance
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
h

Ckernel
Dbias
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
R
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
Ќ
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
h

Vkernel
Wbias
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
 
д
0
1
2
3
(4
)5
*6
+7
08
19
;10
<11
=12
>13
C14
D15
N16
O17
P18
Q19
V20
W21
v
0
1
2
3
(4
)5
06
17
;8
<9
C10
D11
N12
O13
V14
W15
Г
\non_trainable_variables

]layers
^layer_regularization_losses
_metrics
regularization_losses
	variables
`layer_metrics
trainable_variables
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
anon_trainable_variables
	variables
blayer_regularization_losses
cmetrics
regularization_losses

dlayers
elayer_metrics
trainable_variables
 
 
 
Г
fnon_trainable_variables
	variables
glayer_regularization_losses
hmetrics
regularization_losses

ilayers
jlayer_metrics
trainable_variables
YW
VARIABLE_VALUEcovtr2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
knon_trainable_variables
	variables
llayer_regularization_losses
mmetrics
 regularization_losses

nlayers
olayer_metrics
!trainable_variables
 
 
 
Г
pnon_trainable_variables
#	variables
qlayer_regularization_losses
rmetrics
$regularization_losses

slayers
tlayer_metrics
%trainable_variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
*2
+3
 

(0
)1
Г
unon_trainable_variables
,	variables
vlayer_regularization_losses
wmetrics
-regularization_losses

xlayers
ylayer_metrics
.trainable_variables
YW
VARIABLE_VALUEcovtr3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
Г
znon_trainable_variables
2	variables
{layer_regularization_losses
|metrics
3regularization_losses

}layers
~layer_metrics
4trainable_variables
 
 
 
▒
non_trainable_variables
6	variables
 ђlayer_regularization_losses
Ђmetrics
7regularization_losses
ѓlayers
Ѓlayer_metrics
8trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

;0
<1
=2
>3
 

;0
<1
▓
ёnon_trainable_variables
?	variables
 Ёlayer_regularization_losses
єmetrics
@regularization_losses
Єlayers
ѕlayer_metrics
Atrainable_variables
YW
VARIABLE_VALUEcovtr4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

C0
D1
 

C0
D1
▓
Ѕnon_trainable_variables
E	variables
 іlayer_regularization_losses
Іmetrics
Fregularization_losses
їlayers
Їlayer_metrics
Gtrainable_variables
 
 
 
▓
јnon_trainable_variables
I	variables
 Јlayer_regularization_losses
љmetrics
Jregularization_losses
Љlayers
њlayer_metrics
Ktrainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

N0
O1
P2
Q3
 

N0
O1
▓
Њnon_trainable_variables
R	variables
 ћlayer_regularization_losses
Ћmetrics
Sregularization_losses
ќlayers
Ќlayer_metrics
Ttrainable_variables
WU
VARIABLE_VALUEcov3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	cov3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

V0
W1
 

V0
W1
▓
ўnon_trainable_variables
X	variables
 Ўlayer_regularization_losses
џmetrics
Yregularization_losses
Џlayers
юlayer_metrics
Ztrainable_variables
*
*0
+1
=2
>3
P4
Q5
^
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
*0
+1
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
=0
>1
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
P0
Q1
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
з
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
GPU2 *0J 8ѓ *,
f'R%
#__inference_signature_wrapper_28406
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
З	
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
GPU2 *0J 8ѓ *'
f"R 
__inference__traced_save_29146
и
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
GPU2 *0J 8ѓ **
f%R#
!__inference__traced_restore_29222ой
├
{
&__inference_covtr3_layer_call_fn_27557

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *J
fERC
A__inference_covtr3_layer_call_and_return_conditional_losses_275472
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           22

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
┐
y
$__inference_cov3_layer_call_fn_27858

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallј
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
GPU2 *0J 8ѓ *H
fCRA
?__inference_cov3_layer_call_and_return_conditional_losses_278482
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           P::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           P
 
_user_specified_nameinputs
Щ=
┼
D__inference_Generator_layer_call_and_return_conditional_losses_28308

inputs
dense_28251
dense_28253
covtr2_28257
covtr2_28259
batch_normalization_28263
batch_normalization_28265
batch_normalization_28267
batch_normalization_28269
covtr3_28272
covtr3_28274
batch_normalization_1_28278
batch_normalization_1_28280
batch_normalization_1_28282
batch_normalization_1_28284
covtr4_28287
covtr4_28289
batch_normalization_2_28293
batch_normalization_2_28295
batch_normalization_2_28297
batch_normalization_2_28299

cov3_28302

cov3_28304
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallбcov3/StatefulPartitionedCallбcovtr2/StatefulPartitionedCallбcovtr3/StatefulPartitionedCallбcovtr4/StatefulPartitionedCallбdense/StatefulPartitionedCallѕ
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_28251dense_28253*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Г*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_278732
dense/StatefulPartitionedCall§
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
GPU2 *0J 8ѓ *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_279032
reshape/PartitionedCall└
covtr2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr2_28257covtr2_28259*
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
GPU2 *0J 8ѓ *J
fERC
A__inference_covtr2_layer_call_and_return_conditional_losses_273992 
covtr2/StatefulPartitionedCallю
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
GPU2 *0J 8ѓ *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_279212
leaky_re_lu/PartitionedCall┐
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_28263batch_normalization_28265batch_normalization_28267batch_normalization_28269*
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
GPU2 *0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_275022-
+batch_normalization/StatefulPartitionedCallн
covtr3/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr3_28272covtr3_28274*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *J
fERC
A__inference_covtr3_layer_call_and_return_conditional_losses_275472 
covtr3/StatefulPartitionedCallб
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_279742
leaky_re_lu_1/PartitionedCall¤
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_28278batch_normalization_1_28280batch_normalization_1_28282batch_normalization_1_28284*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           2*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_276502/
-batch_normalization_1/StatefulPartitionedCallо
covtr4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr4_28287covtr4_28289*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *J
fERC
A__inference_covtr4_layer_call_and_return_conditional_losses_276952 
covtr4/StatefulPartitionedCallб
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_280272
leaky_re_lu_2/PartitionedCall¤
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_28293batch_normalization_2_28295batch_normalization_2_28297batch_normalization_2_28299*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           P*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_277982/
-batch_normalization_2/StatefulPartitionedCall╠
cov3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0
cov3_28302
cov3_28304*
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
GPU2 *0J 8ѓ *H
fCRA
?__inference_cov3_layer_call_and_return_conditional_losses_278482
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
ѕ
I
-__inference_leaky_re_lu_2_layer_call_fn_28993

inputs
identityт
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_280272
PartitionedCallє
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           P2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           P:i e
A
_output_shapes/
-:+                           P
 
_user_specified_nameinputs
ъ
д
3__inference_batch_normalization_layer_call_fn_28909

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallи
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
GPU2 *0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_275022
StatefulPartitionedCallе
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
├
{
&__inference_covtr2_layer_call_fn_27409

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallљ
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
GPU2 *0J 8ѓ *J
fERC
A__inference_covtr2_layer_call_and_return_conditional_losses_273992
StatefulPartitionedCallе
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
Ѓ>
╚
D__inference_Generator_layer_call_and_return_conditional_losses_28136
	gen_noise
dense_28079
dense_28081
covtr2_28085
covtr2_28087
batch_normalization_28091
batch_normalization_28093
batch_normalization_28095
batch_normalization_28097
covtr3_28100
covtr3_28102
batch_normalization_1_28106
batch_normalization_1_28108
batch_normalization_1_28110
batch_normalization_1_28112
covtr4_28115
covtr4_28117
batch_normalization_2_28121
batch_normalization_2_28123
batch_normalization_2_28125
batch_normalization_2_28127

cov3_28130

cov3_28132
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallбcov3/StatefulPartitionedCallбcovtr2/StatefulPartitionedCallбcovtr3/StatefulPartitionedCallбcovtr4/StatefulPartitionedCallбdense/StatefulPartitionedCallІ
dense/StatefulPartitionedCallStatefulPartitionedCall	gen_noisedense_28079dense_28081*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Г*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_278732
dense/StatefulPartitionedCall§
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
GPU2 *0J 8ѓ *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_279032
reshape/PartitionedCall└
covtr2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr2_28085covtr2_28087*
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
GPU2 *0J 8ѓ *J
fERC
A__inference_covtr2_layer_call_and_return_conditional_losses_273992 
covtr2/StatefulPartitionedCallю
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
GPU2 *0J 8ѓ *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_279212
leaky_re_lu/PartitionedCall┐
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_28091batch_normalization_28093batch_normalization_28095batch_normalization_28097*
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
GPU2 *0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_275022-
+batch_normalization/StatefulPartitionedCallн
covtr3/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr3_28100covtr3_28102*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *J
fERC
A__inference_covtr3_layer_call_and_return_conditional_losses_275472 
covtr3/StatefulPartitionedCallб
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_279742
leaky_re_lu_1/PartitionedCall¤
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_28106batch_normalization_1_28108batch_normalization_1_28110batch_normalization_1_28112*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           2*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_276502/
-batch_normalization_1/StatefulPartitionedCallо
covtr4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr4_28115covtr4_28117*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *J
fERC
A__inference_covtr4_layer_call_and_return_conditional_losses_276952 
covtr4/StatefulPartitionedCallб
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_280272
leaky_re_lu_2/PartitionedCall¤
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_28121batch_normalization_2_28123batch_normalization_2_28125batch_normalization_2_28127*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           P*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_277982/
-batch_normalization_2/StatefulPartitionedCall╠
cov3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0
cov3_28130
cov3_28132*
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
GPU2 *0J 8ѓ *H
fCRA
?__inference_cov3_layer_call_and_return_conditional_losses_278482
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
ќ
Є
N__inference_batch_normalization_layer_call_and_return_conditional_losses_28883

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
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
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
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
ѕ
I
-__inference_leaky_re_lu_1_layer_call_fn_28919

inputs
identityт
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_279742
PartitionedCallє
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           2:i e
A
_output_shapes/
-:+                           2
 
_user_specified_nameinputs
»
И
)__inference_Generator_layer_call_fn_28246
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
identityѕбStatefulPartitionedCallЪ
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
GPU2 *0J 8ѓ *M
fHRF
D__inference_Generator_layer_call_and_return_conditional_losses_281992
StatefulPartitionedCallе
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
ї7
ю

__inference__traced_save_29146
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

identity_1ѕбMergeV2CheckpointsЈ
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
ConstЇ
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_ce6c2b10fd4e4b0e8c2249466451df0f/part2	
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameѕ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*џ

valueљ
BЇ
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesХ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЕ

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
&MergeV2Checkpoints/checkpoint_prefixesА
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

identity_1Identity_1:output:0*М
_input_shapes┴
Й: :	dГ:Г: : : : : : :2 :2:2:2:2:2:P2:P:P:P:P:P:P:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	dГ:!

_output_shapes	
:Г:,(
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
:2 : 


_output_shapes
:2: 

_output_shapes
:2: 

_output_shapes
:2: 

_output_shapes
:2: 

_output_shapes
:2:,(
&
_output_shapes
:P2: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P:,(
&
_output_shapes
:P: 

_output_shapes
::

_output_shapes
: 
б
е
5__inference_batch_normalization_2_layer_call_fn_29057

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           P*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_277982
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           P2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           P::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           P
 
_user_specified_nameinputs
х
И
)__inference_Generator_layer_call_fn_28355
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
identityѕбStatefulPartitionedCallЦ
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
GPU2 *0J 8ѓ *M
fHRF
D__inference_Generator_layer_call_and_return_conditional_losses_283082
StatefulPartitionedCallе
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
ў
Ѕ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27650

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:2*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:2*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:2*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:2*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           2:2:2:2:2:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           22

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           2:::::i e
A
_output_shapes/
-:+                           2
 
_user_specified_nameinputs
ф
d
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_28914

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           22
	LeakyReluЁ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           2:i e
A
_output_shapes/
-:+                           2
 
_user_specified_nameinputs
Ы!
│
A__inference_covtr4_layer_call_and_return_conditional_losses_27695

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityѕD
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
strided_slice/stack_2Р
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
strided_slice_1/stack_2В
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
strided_slice_2/stack_2В
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
value	B :P2	
stack/3ѓ
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
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:P2*
dtype02!
conv2d_transpose/ReadVariableOp­
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           P*
paddingSAME*
strides
2
conv2d_transposeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpц
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           P2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           P2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           2:::i e
A
_output_shapes/
-:+                           2
 
_user_specified_nameinputs
╚
Г
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27619

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбAssignNewValueбAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:2*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:2*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:2*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:2*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           2:2:2:2:2:*
epsilon%oЃ:*
exponential_avg_factor%═╠L>2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЇ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1д
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           22

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           2::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           2
 
_user_specified_nameinputs
у
▓
#__inference_signature_wrapper_28406
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
identityѕбStatefulPartitionedCall№
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
GPU2 *0J 8ѓ *)
f$R"
 __inference__wrapped_model_273652
StatefulPartitionedCallќ
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
Ц
C
'__inference_reshape_layer_call_fn_28835

inputs
identity═
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
GPU2 *0J 8ѓ *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_279032
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*'
_input_shapes
:         Г:P L
(
_output_shapes
:         Г
 
_user_specified_nameinputs
ф
d
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_28027

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           P2
	LeakyReluЁ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           P2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           P:i e
A
_output_shapes/
-:+                           P
 
_user_specified_nameinputs
Ф│
Ы
D__inference_Generator_layer_call_and_return_conditional_losses_28555

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
identityѕб"batch_normalization/AssignNewValueб$batch_normalization/AssignNewValue_1б$batch_normalization_1/AssignNewValueб&batch_normalization_1/AssignNewValue_1б$batch_normalization_2/AssignNewValueб&batch_normalization_2/AssignNewValue_1а
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	dГ*
dtype02
dense/MatMul/ReadVariableOpє
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Г2
dense/MatMulЪ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:Г*
dtype02
dense/BiasAdd/ReadVariableOpџ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Г2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:         Г2

dense/Reluf
reshape/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shapeё
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackѕ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1ѕ
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2њ
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
reshape/Reshape/shape/3Ж
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeА
reshape/ReshapeReshapedense/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:         2
reshape/Reshaped
covtr2/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
covtr2/Shapeѓ
covtr2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr2/strided_slice/stackє
covtr2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr2/strided_slice/stack_1є
covtr2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr2/strided_slice/stack_2ї
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
covtr2/stackє
covtr2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr2/strided_slice_1/stackі
covtr2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr2/strided_slice_1/stack_1і
covtr2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr2/strided_slice_1/stack_2ќ
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
&covtr2/conv2d_transpose/ReadVariableOpї
covtr2/conv2d_transposeConv2DBackpropInputcovtr2/stack:output:0.covtr2/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
covtr2/conv2d_transposeА
covtr2/BiasAdd/ReadVariableOpReadVariableOp&covtr2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
covtr2/BiasAdd/ReadVariableOp«
covtr2/BiasAddBiasAdd covtr2/conv2d_transpose:output:0%covtr2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
covtr2/BiasAddЁ
leaky_re_lu/LeakyRelu	LeakyRelucovtr2/BiasAdd:output:0*/
_output_shapes
:          2
leaky_re_lu/LeakyRelu░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOpХ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1с
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpж
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ь
$batch_normalization/FusedBatchNormV3FusedBatchNormV3#leaky_re_lu/LeakyRelu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
exponential_avg_factor%═╠L>2&
$batch_normalization/FusedBatchNormV3э
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValueЁ
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1t
covtr3/ShapeShape(batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr3/Shapeѓ
covtr3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr3/strided_slice/stackє
covtr3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr3/strided_slice/stack_1є
covtr3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr3/strided_slice/stack_2ї
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
value	B :22
covtr3/stack/3╝
covtr3/stackPackcovtr3/strided_slice:output:0covtr3/stack/1:output:0covtr3/stack/2:output:0covtr3/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr3/stackє
covtr3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr3/strided_slice_1/stackі
covtr3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr3/strided_slice_1/stack_1і
covtr3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr3/strided_slice_1/stack_2ќ
covtr3/strided_slice_1StridedSlicecovtr3/stack:output:0%covtr3/strided_slice_1/stack:output:0'covtr3/strided_slice_1/stack_1:output:0'covtr3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr3/strided_slice_1╚
&covtr3/conv2d_transpose/ReadVariableOpReadVariableOp/covtr3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:2 *
dtype02(
&covtr3/conv2d_transpose/ReadVariableOpю
covtr3/conv2d_transposeConv2DBackpropInputcovtr3/stack:output:0.covtr3/conv2d_transpose/ReadVariableOp:value:0(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         4,2*
paddingSAME*
strides
2
covtr3/conv2d_transposeА
covtr3/BiasAdd/ReadVariableOpReadVariableOp&covtr3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
covtr3/BiasAdd/ReadVariableOp«
covtr3/BiasAddBiasAdd covtr3/conv2d_transpose:output:0%covtr3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         4,22
covtr3/BiasAddЅ
leaky_re_lu_1/LeakyRelu	LeakyRelucovtr3/BiasAdd:output:0*/
_output_shapes
:         4,22
leaky_re_lu_1/LeakyReluХ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:2*
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:2*
dtype02(
&batch_normalization_1/ReadVariableOp_1ж
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:2*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:2*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ч
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_1/LeakyRelu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         4,2:2:2:2:2:*
epsilon%oЃ:*
exponential_avg_factor%═╠L>2(
&batch_normalization_1/FusedBatchNormV3Ѓ
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValueЉ
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1v
covtr4/ShapeShape*batch_normalization_1/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr4/Shapeѓ
covtr4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr4/strided_slice/stackє
covtr4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr4/strided_slice/stack_1є
covtr4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr4/strided_slice/stack_2ї
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
covtr4/stack/2b
covtr4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :P2
covtr4/stack/3╝
covtr4/stackPackcovtr4/strided_slice:output:0covtr4/stack/1:output:0covtr4/stack/2:output:0covtr4/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr4/stackє
covtr4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr4/strided_slice_1/stackі
covtr4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr4/strided_slice_1/stack_1і
covtr4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr4/strided_slice_1/stack_2ќ
covtr4/strided_slice_1StridedSlicecovtr4/stack:output:0%covtr4/strided_slice_1/stack:output:0'covtr4/strided_slice_1/stack_1:output:0'covtr4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr4/strided_slice_1╚
&covtr4/conv2d_transpose/ReadVariableOpReadVariableOp/covtr4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:P2*
dtype02(
&covtr4/conv2d_transpose/ReadVariableOpъ
covtr4/conv2d_transposeConv2DBackpropInputcovtr4/stack:output:0.covtr4/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         hXP*
paddingSAME*
strides
2
covtr4/conv2d_transposeА
covtr4/BiasAdd/ReadVariableOpReadVariableOp&covtr4_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
covtr4/BiasAdd/ReadVariableOp«
covtr4/BiasAddBiasAdd covtr4/conv2d_transpose:output:0%covtr4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         hXP2
covtr4/BiasAddЅ
leaky_re_lu_2/LeakyRelu	LeakyRelucovtr4/BiasAdd:output:0*/
_output_shapes
:         hXP2
leaky_re_lu_2/LeakyReluХ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:P*
dtype02&
$batch_normalization_2/ReadVariableOp╝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:P*
dtype02(
&batch_normalization_2/ReadVariableOp_1ж
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ч
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_2/LeakyRelu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         hXP:P:P:P:P:*
epsilon%oЃ:*
exponential_avg_factor%═╠L>2(
&batch_normalization_2/FusedBatchNormV3Ѓ
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValueЉ
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1r

cov3/ShapeShape*batch_normalization_2/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2

cov3/Shape~
cov3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cov3/strided_slice/stackѓ
cov3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice/stack_1ѓ
cov3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice/stack_2ђ
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

cov3/stackѓ
cov3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cov3/strided_slice_1/stackє
cov3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice_1/stack_1є
cov3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice_1/stack_2і
cov3/strided_slice_1StridedSlicecov3/stack:output:0#cov3/strided_slice_1/stack:output:0%cov3/strided_slice_1/stack_1:output:0%cov3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cov3/strided_slice_1┬
$cov3/conv2d_transpose/ReadVariableOpReadVariableOp-cov3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:P*
dtype02&
$cov3/conv2d_transpose/ReadVariableOpЌ
cov3/conv2d_transposeConv2DBackpropInputcov3/stack:output:0,cov3/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         mY*
paddingVALID*
strides
2
cov3/conv2d_transposeЏ
cov3/BiasAdd/ReadVariableOpReadVariableOp$cov3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
cov3/BiasAdd/ReadVariableOpд
cov3/BiasAddBiasAddcov3/conv2d_transpose:output:0#cov3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         mY2
cov3/BiasAddx
cov3/SigmoidSigmoidcov3/BiasAdd:output:0*
T0*/
_output_shapes
:         mY2
cov3/Sigmoidп
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
ю
д
3__inference_batch_normalization_layer_call_fn_28896

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallх
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
GPU2 *0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_274712
StatefulPartitionedCallе
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
╚
Г
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_29013

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбAssignNewValueбAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           P:P:P:P:P:*
epsilon%oЃ:*
exponential_avg_factor%═╠L>2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЇ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1д
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           P2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           P::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           P
 
_user_specified_nameinputs
ќ
Є
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27502

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
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
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
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
Ы!
│
A__inference_covtr3_layer_call_and_return_conditional_losses_27547

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityѕD
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
strided_slice/stack_2Р
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
strided_slice_1/stack_2В
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
strided_slice_2/stack_2В
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
value	B :22	
stack/3ѓ
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
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:2 *
dtype02!
conv2d_transpose/ReadVariableOp­
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           2*
paddingSAME*
strides
2
conv2d_transposeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpц
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           22	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           22

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            :::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
д
х
)__inference_Generator_layer_call_fn_28747

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
identityѕбStatefulPartitionedCallю
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
GPU2 *0J 8ѓ *M
fHRF
D__inference_Generator_layer_call_and_return_conditional_losses_281992
StatefulPartitionedCallе
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
Ф
е
@__inference_dense_layer_call_and_return_conditional_losses_28807

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	dГ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Г2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Г*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Г2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Г2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         Г2

Identity"
identityIdentity:output:0*.
_input_shapes
:         d:::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
З=
┼
D__inference_Generator_layer_call_and_return_conditional_losses_28199

inputs
dense_28142
dense_28144
covtr2_28148
covtr2_28150
batch_normalization_28154
batch_normalization_28156
batch_normalization_28158
batch_normalization_28160
covtr3_28163
covtr3_28165
batch_normalization_1_28169
batch_normalization_1_28171
batch_normalization_1_28173
batch_normalization_1_28175
covtr4_28178
covtr4_28180
batch_normalization_2_28184
batch_normalization_2_28186
batch_normalization_2_28188
batch_normalization_2_28190

cov3_28193

cov3_28195
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallбcov3/StatefulPartitionedCallбcovtr2/StatefulPartitionedCallбcovtr3/StatefulPartitionedCallбcovtr4/StatefulPartitionedCallбdense/StatefulPartitionedCallѕ
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_28142dense_28144*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Г*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_278732
dense/StatefulPartitionedCall§
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
GPU2 *0J 8ѓ *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_279032
reshape/PartitionedCall└
covtr2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr2_28148covtr2_28150*
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
GPU2 *0J 8ѓ *J
fERC
A__inference_covtr2_layer_call_and_return_conditional_losses_273992 
covtr2/StatefulPartitionedCallю
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
GPU2 *0J 8ѓ *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_279212
leaky_re_lu/PartitionedCallй
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_28154batch_normalization_28156batch_normalization_28158batch_normalization_28160*
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
GPU2 *0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_274712-
+batch_normalization/StatefulPartitionedCallн
covtr3/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr3_28163covtr3_28165*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *J
fERC
A__inference_covtr3_layer_call_and_return_conditional_losses_275472 
covtr3/StatefulPartitionedCallб
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_279742
leaky_re_lu_1/PartitionedCall═
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_28169batch_normalization_1_28171batch_normalization_1_28173batch_normalization_1_28175*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_276192/
-batch_normalization_1/StatefulPartitionedCallо
covtr4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr4_28178covtr4_28180*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *J
fERC
A__inference_covtr4_layer_call_and_return_conditional_losses_276952 
covtr4/StatefulPartitionedCallб
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_280272
leaky_re_lu_2/PartitionedCall═
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_28184batch_normalization_2_28186batch_normalization_2_28188batch_normalization_2_28190*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_277672/
-batch_normalization_2/StatefulPartitionedCall╠
cov3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0
cov3_28193
cov3_28195*
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
GPU2 *0J 8ѓ *H
fCRA
?__inference_cov3_layer_call_and_return_conditional_losses_278482
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
г
х
)__inference_Generator_layer_call_fn_28796

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
identityѕбStatefulPartitionedCallб
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
GPU2 *0J 8ѓ *M
fHRF
D__inference_Generator_layer_call_and_return_conditional_losses_283082
StatefulPartitionedCallе
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
Ы!
│
A__inference_covtr2_layer_call_and_return_conditional_losses_27399

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityѕD
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
strided_slice/stack_2Р
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
strided_slice_1/stack_2В
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
strided_slice_2/stack_2В
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
stack/3ѓ
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
strided_slice_3/stack_2В
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
conv2d_transpose/ReadVariableOp­
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
conv2d_transposeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpц
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
Ф
е
@__inference_dense_layer_call_and_return_conditional_losses_27873

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	dГ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Г2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Г*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Г2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Г2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         Г2

Identity"
identityIdentity:output:0*.
_input_shapes
:         d:::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
е
b
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_28840

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                            2
	LeakyReluЁ
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
е
b
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_27921

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                            2
	LeakyReluЁ
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
к
Ф
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27471

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбAssignNewValueбAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
exponential_avg_factor%═╠L>2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЇ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1д
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
█
z
%__inference_dense_layer_call_fn_28816

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Г*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_278732
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         Г2

Identity"
identityIdentity:output:0*.
_input_shapes
:         d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
├
{
&__inference_covtr4_layer_call_fn_27705

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *J
fERC
A__inference_covtr4_layer_call_and_return_conditional_losses_276952
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           P2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           2::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           2
 
_user_specified_nameinputs
џЎ
є

D__inference_Generator_layer_call_and_return_conditional_losses_28698

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
identityѕа
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	dГ*
dtype02
dense/MatMul/ReadVariableOpє
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Г2
dense/MatMulЪ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:Г*
dtype02
dense/BiasAdd/ReadVariableOpџ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Г2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:         Г2

dense/Reluf
reshape/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shapeё
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackѕ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1ѕ
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2њ
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
reshape/Reshape/shape/3Ж
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeА
reshape/ReshapeReshapedense/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:         2
reshape/Reshaped
covtr2/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
covtr2/Shapeѓ
covtr2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr2/strided_slice/stackє
covtr2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr2/strided_slice/stack_1є
covtr2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr2/strided_slice/stack_2ї
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
covtr2/stackє
covtr2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr2/strided_slice_1/stackі
covtr2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr2/strided_slice_1/stack_1і
covtr2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr2/strided_slice_1/stack_2ќ
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
&covtr2/conv2d_transpose/ReadVariableOpї
covtr2/conv2d_transposeConv2DBackpropInputcovtr2/stack:output:0.covtr2/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
covtr2/conv2d_transposeА
covtr2/BiasAdd/ReadVariableOpReadVariableOp&covtr2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
covtr2/BiasAdd/ReadVariableOp«
covtr2/BiasAddBiasAdd covtr2/conv2d_transpose:output:0%covtr2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
covtr2/BiasAddЁ
leaky_re_lu/LeakyRelu	LeakyRelucovtr2/BiasAdd:output:0*/
_output_shapes
:          2
leaky_re_lu/LeakyRelu░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOpХ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1с
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpж
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
epsilon%oЃ:*
is_training( 2&
$batch_normalization/FusedBatchNormV3t
covtr3/ShapeShape(batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr3/Shapeѓ
covtr3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr3/strided_slice/stackє
covtr3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr3/strided_slice/stack_1є
covtr3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr3/strided_slice/stack_2ї
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
value	B :22
covtr3/stack/3╝
covtr3/stackPackcovtr3/strided_slice:output:0covtr3/stack/1:output:0covtr3/stack/2:output:0covtr3/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr3/stackє
covtr3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr3/strided_slice_1/stackі
covtr3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr3/strided_slice_1/stack_1і
covtr3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr3/strided_slice_1/stack_2ќ
covtr3/strided_slice_1StridedSlicecovtr3/stack:output:0%covtr3/strided_slice_1/stack:output:0'covtr3/strided_slice_1/stack_1:output:0'covtr3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr3/strided_slice_1╚
&covtr3/conv2d_transpose/ReadVariableOpReadVariableOp/covtr3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:2 *
dtype02(
&covtr3/conv2d_transpose/ReadVariableOpю
covtr3/conv2d_transposeConv2DBackpropInputcovtr3/stack:output:0.covtr3/conv2d_transpose/ReadVariableOp:value:0(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         4,2*
paddingSAME*
strides
2
covtr3/conv2d_transposeА
covtr3/BiasAdd/ReadVariableOpReadVariableOp&covtr3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
covtr3/BiasAdd/ReadVariableOp«
covtr3/BiasAddBiasAdd covtr3/conv2d_transpose:output:0%covtr3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         4,22
covtr3/BiasAddЅ
leaky_re_lu_1/LeakyRelu	LeakyRelucovtr3/BiasAdd:output:0*/
_output_shapes
:         4,22
leaky_re_lu_1/LeakyReluХ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:2*
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:2*
dtype02(
&batch_normalization_1/ReadVariableOp_1ж
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:2*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:2*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ь
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_1/LeakyRelu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         4,2:2:2:2:2:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3v
covtr4/ShapeShape*batch_normalization_1/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr4/Shapeѓ
covtr4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr4/strided_slice/stackє
covtr4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr4/strided_slice/stack_1є
covtr4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr4/strided_slice/stack_2ї
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
covtr4/stack/2b
covtr4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :P2
covtr4/stack/3╝
covtr4/stackPackcovtr4/strided_slice:output:0covtr4/stack/1:output:0covtr4/stack/2:output:0covtr4/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr4/stackє
covtr4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr4/strided_slice_1/stackі
covtr4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr4/strided_slice_1/stack_1і
covtr4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr4/strided_slice_1/stack_2ќ
covtr4/strided_slice_1StridedSlicecovtr4/stack:output:0%covtr4/strided_slice_1/stack:output:0'covtr4/strided_slice_1/stack_1:output:0'covtr4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr4/strided_slice_1╚
&covtr4/conv2d_transpose/ReadVariableOpReadVariableOp/covtr4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:P2*
dtype02(
&covtr4/conv2d_transpose/ReadVariableOpъ
covtr4/conv2d_transposeConv2DBackpropInputcovtr4/stack:output:0.covtr4/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         hXP*
paddingSAME*
strides
2
covtr4/conv2d_transposeА
covtr4/BiasAdd/ReadVariableOpReadVariableOp&covtr4_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
covtr4/BiasAdd/ReadVariableOp«
covtr4/BiasAddBiasAdd covtr4/conv2d_transpose:output:0%covtr4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         hXP2
covtr4/BiasAddЅ
leaky_re_lu_2/LeakyRelu	LeakyRelucovtr4/BiasAdd:output:0*/
_output_shapes
:         hXP2
leaky_re_lu_2/LeakyReluХ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:P*
dtype02&
$batch_normalization_2/ReadVariableOp╝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:P*
dtype02(
&batch_normalization_2/ReadVariableOp_1ж
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ь
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_2/LeakyRelu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         hXP:P:P:P:P:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3r

cov3/ShapeShape*batch_normalization_2/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2

cov3/Shape~
cov3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cov3/strided_slice/stackѓ
cov3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice/stack_1ѓ
cov3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice/stack_2ђ
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

cov3/stackѓ
cov3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cov3/strided_slice_1/stackє
cov3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice_1/stack_1є
cov3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice_1/stack_2і
cov3/strided_slice_1StridedSlicecov3/stack:output:0#cov3/strided_slice_1/stack:output:0%cov3/strided_slice_1/stack_1:output:0%cov3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cov3/strided_slice_1┬
$cov3/conv2d_transpose/ReadVariableOpReadVariableOp-cov3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:P*
dtype02&
$cov3/conv2d_transpose/ReadVariableOpЌ
cov3/conv2d_transposeConv2DBackpropInputcov3/stack:output:0,cov3/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         mY*
paddingVALID*
strides
2
cov3/conv2d_transposeЏ
cov3/BiasAdd/ReadVariableOpReadVariableOp$cov3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
cov3/BiasAdd/ReadVariableOpд
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
╚
Г
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_28939

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбAssignNewValueбAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:2*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:2*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:2*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:2*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           2:2:2:2:2:*
epsilon%oЃ:*
exponential_avg_factor%═╠L>2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЇ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1д
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           22

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           2::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           2
 
_user_specified_nameinputs
ў
Ѕ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_27798

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           P:P:P:P:P:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           P2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           P:::::i e
A
_output_shapes/
-:+                           P
 
_user_specified_nameinputs
а
е
5__inference_batch_normalization_2_layer_call_fn_29044

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_277672
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           P2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           P::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           P
 
_user_specified_nameinputs
и%
▒
?__inference_cov3_layer_call_and_return_conditional_losses_27848

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityѕD
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
strided_slice/stack_2Р
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
strided_slice_1/stack_2В
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
strided_slice_2/stack_2В
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
stack/3ѓ
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
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:P*
dtype02!
conv2d_transpose/ReadVariableOpы
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
conv2d_transposeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpц
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
identityIdentity:output:0*H
_input_shapes7
5:+                           P:::i e
A
_output_shapes/
-:+                           P
 
_user_specified_nameinputs
т
^
B__inference_reshape_layer_call_and_return_conditional_losses_27903

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
strided_slice/stack_2Р
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
:         Г:P L
(
_output_shapes
:         Г
 
_user_specified_nameinputs
ў
Ѕ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_28957

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:2*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:2*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:2*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:2*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           2:2:2:2:2:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           22

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           2:::::i e
A
_output_shapes/
-:+                           2
 
_user_specified_nameinputs
к
Ф
N__inference_batch_normalization_layer_call_and_return_conditional_losses_28865

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбAssignNewValueбAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
exponential_avg_factor%═╠L>2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЇ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1д
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
§=
╚
D__inference_Generator_layer_call_and_return_conditional_losses_28076
	gen_noise
dense_27884
dense_27886
covtr2_27911
covtr2_27913
batch_normalization_27955
batch_normalization_27957
batch_normalization_27959
batch_normalization_27961
covtr3_27964
covtr3_27966
batch_normalization_1_28008
batch_normalization_1_28010
batch_normalization_1_28012
batch_normalization_1_28014
covtr4_28017
covtr4_28019
batch_normalization_2_28061
batch_normalization_2_28063
batch_normalization_2_28065
batch_normalization_2_28067

cov3_28070

cov3_28072
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallбcov3/StatefulPartitionedCallбcovtr2/StatefulPartitionedCallбcovtr3/StatefulPartitionedCallбcovtr4/StatefulPartitionedCallбdense/StatefulPartitionedCallІ
dense/StatefulPartitionedCallStatefulPartitionedCall	gen_noisedense_27884dense_27886*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Г*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_278732
dense/StatefulPartitionedCall§
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
GPU2 *0J 8ѓ *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_279032
reshape/PartitionedCall└
covtr2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr2_27911covtr2_27913*
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
GPU2 *0J 8ѓ *J
fERC
A__inference_covtr2_layer_call_and_return_conditional_losses_273992 
covtr2/StatefulPartitionedCallю
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
GPU2 *0J 8ѓ *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_279212
leaky_re_lu/PartitionedCallй
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_27955batch_normalization_27957batch_normalization_27959batch_normalization_27961*
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
GPU2 *0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_274712-
+batch_normalization/StatefulPartitionedCallн
covtr3/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr3_27964covtr3_27966*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *J
fERC
A__inference_covtr3_layer_call_and_return_conditional_losses_275472 
covtr3/StatefulPartitionedCallб
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_279742
leaky_re_lu_1/PartitionedCall═
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_28008batch_normalization_1_28010batch_normalization_1_28012batch_normalization_1_28014*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_276192/
-batch_normalization_1/StatefulPartitionedCallо
covtr4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr4_28017covtr4_28019*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *J
fERC
A__inference_covtr4_layer_call_and_return_conditional_losses_276952 
covtr4/StatefulPartitionedCallб
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_280272
leaky_re_lu_2/PartitionedCall═
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_28061batch_normalization_2_28063batch_normalization_2_28065batch_normalization_2_28067*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_277672/
-batch_normalization_2/StatefulPartitionedCall╠
cov3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0
cov3_28070
cov3_28072*
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
GPU2 *0J 8ѓ *H
fCRA
?__inference_cov3_layer_call_and_return_conditional_losses_278482
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
ф
d
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_27974

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           22
	LeakyReluЁ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           2:i e
A
_output_shapes/
-:+                           2
 
_user_specified_nameinputs
т
^
B__inference_reshape_layer_call_and_return_conditional_losses_28830

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
strided_slice/stack_2Р
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
:         Г:P L
(
_output_shapes
:         Г
 
_user_specified_nameinputs
│_
Ф
!__inference__traced_restore_29222
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
identity_23ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9ј
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*џ

valueљ
BЇ
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names╝
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesъ
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

Identityю
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1б
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ц
AssignVariableOp_2AssignVariableOp assignvariableop_2_covtr2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Б
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

Identity_6и
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

Identity_8Ц
AssignVariableOp_8AssignVariableOp assignvariableop_8_covtr3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Б
AssignVariableOp_9AssignVariableOpassignvariableop_9_covtr3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10и
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_1_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Х
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_1_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12й
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
Identity_14Е
AssignVariableOp_14AssignVariableOp!assignvariableop_14_covtr4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Д
AssignVariableOp_15AssignVariableOpassignvariableop_15_covtr4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16и
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_2_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Х
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_2_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18й
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
Identity_20Д
AssignVariableOp_20AssignVariableOpassignvariableop_20_cov3_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ц
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
Identity_22х
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
а
е
5__inference_batch_normalization_1_layer_call_fn_28970

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_276192
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           22

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           2::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           2
 
_user_specified_nameinputs
У┤
┴
 __inference__wrapped_model_27365
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
identityѕЙ
%Generator/dense/MatMul/ReadVariableOpReadVariableOp.generator_dense_matmul_readvariableop_resource*
_output_shapes
:	dГ*
dtype02'
%Generator/dense/MatMul/ReadVariableOpД
Generator/dense/MatMulMatMul	gen_noise-Generator/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Г2
Generator/dense/MatMulй
&Generator/dense/BiasAdd/ReadVariableOpReadVariableOp/generator_dense_biasadd_readvariableop_resource*
_output_shapes	
:Г*
dtype02(
&Generator/dense/BiasAdd/ReadVariableOp┬
Generator/dense/BiasAddBiasAdd Generator/dense/MatMul:product:0.Generator/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Г2
Generator/dense/BiasAddЅ
Generator/dense/ReluRelu Generator/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         Г2
Generator/dense/Reluё
Generator/reshape/ShapeShape"Generator/dense/Relu:activations:0*
T0*
_output_shapes
:2
Generator/reshape/Shapeў
%Generator/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Generator/reshape/strided_slice/stackю
'Generator/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/reshape/strided_slice/stack_1ю
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
Generator/reshape/strided_sliceѕ
!Generator/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!Generator/reshape/Reshape/shape/1ѕ
!Generator/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!Generator/reshape/Reshape/shape/2ѕ
!Generator/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!Generator/reshape/Reshape/shape/3д
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
Generator/reshape/Reshapeѓ
Generator/covtr2/ShapeShape"Generator/reshape/Reshape:output:0*
T0*
_output_shapes
:2
Generator/covtr2/Shapeќ
$Generator/covtr2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr2/strided_slice/stackџ
&Generator/covtr2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr2/strided_slice/stack_1џ
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
Generator/covtr2/stack/3Э
Generator/covtr2/stackPack'Generator/covtr2/strided_slice:output:0!Generator/covtr2/stack/1:output:0!Generator/covtr2/stack/2:output:0!Generator/covtr2/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr2/stackџ
&Generator/covtr2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr2/strided_slice_1/stackъ
(Generator/covtr2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr2/strided_slice_1/stack_1ъ
(Generator/covtr2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr2/strided_slice_1/stack_2м
 Generator/covtr2/strided_slice_1StridedSliceGenerator/covtr2/stack:output:0/Generator/covtr2/strided_slice_1/stack:output:01Generator/covtr2/strided_slice_1/stack_1:output:01Generator/covtr2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr2/strided_slice_1Т
0Generator/covtr2/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype022
0Generator/covtr2/conv2d_transpose/ReadVariableOpЙ
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
'Generator/covtr2/BiasAdd/ReadVariableOpо
Generator/covtr2/BiasAddBiasAdd*Generator/covtr2/conv2d_transpose:output:0/Generator/covtr2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
Generator/covtr2/BiasAddБ
Generator/leaky_re_lu/LeakyRelu	LeakyRelu!Generator/covtr2/BiasAdd:output:0*/
_output_shapes
:          2!
Generator/leaky_re_lu/LeakyRelu╬
,Generator/batch_normalization/ReadVariableOpReadVariableOp5generator_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02.
,Generator/batch_normalization/ReadVariableOpн
.Generator/batch_normalization/ReadVariableOp_1ReadVariableOp7generator_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype020
.Generator/batch_normalization/ReadVariableOp_1Ђ
=Generator/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpFgenerator_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02?
=Generator/batch_normalization/FusedBatchNormV3/ReadVariableOpЄ
?Generator/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHgenerator_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02A
?Generator/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ц
.Generator/batch_normalization/FusedBatchNormV3FusedBatchNormV3-Generator/leaky_re_lu/LeakyRelu:activations:04Generator/batch_normalization/ReadVariableOp:value:06Generator/batch_normalization/ReadVariableOp_1:value:0EGenerator/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0GGenerator/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( 20
.Generator/batch_normalization/FusedBatchNormV3њ
Generator/covtr3/ShapeShape2Generator/batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr3/Shapeќ
$Generator/covtr3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr3/strided_slice/stackџ
&Generator/covtr3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr3/strided_slice/stack_1џ
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
value	B :22
Generator/covtr3/stack/3Э
Generator/covtr3/stackPack'Generator/covtr3/strided_slice:output:0!Generator/covtr3/stack/1:output:0!Generator/covtr3/stack/2:output:0!Generator/covtr3/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr3/stackџ
&Generator/covtr3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr3/strided_slice_1/stackъ
(Generator/covtr3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr3/strided_slice_1/stack_1ъ
(Generator/covtr3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr3/strided_slice_1/stack_2м
 Generator/covtr3/strided_slice_1StridedSliceGenerator/covtr3/stack:output:0/Generator/covtr3/strided_slice_1/stack:output:01Generator/covtr3/strided_slice_1/stack_1:output:01Generator/covtr3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr3/strided_slice_1Т
0Generator/covtr3/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:2 *
dtype022
0Generator/covtr3/conv2d_transpose/ReadVariableOp╬
!Generator/covtr3/conv2d_transposeConv2DBackpropInputGenerator/covtr3/stack:output:08Generator/covtr3/conv2d_transpose/ReadVariableOp:value:02Generator/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         4,2*
paddingSAME*
strides
2#
!Generator/covtr3/conv2d_transpose┐
'Generator/covtr3/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02)
'Generator/covtr3/BiasAdd/ReadVariableOpо
Generator/covtr3/BiasAddBiasAdd*Generator/covtr3/conv2d_transpose:output:0/Generator/covtr3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         4,22
Generator/covtr3/BiasAddД
!Generator/leaky_re_lu_1/LeakyRelu	LeakyRelu!Generator/covtr3/BiasAdd:output:0*/
_output_shapes
:         4,22#
!Generator/leaky_re_lu_1/LeakyReluн
.Generator/batch_normalization_1/ReadVariableOpReadVariableOp7generator_batch_normalization_1_readvariableop_resource*
_output_shapes
:2*
dtype020
.Generator/batch_normalization_1/ReadVariableOp┌
0Generator/batch_normalization_1/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:2*
dtype022
0Generator/batch_normalization_1/ReadVariableOp_1Є
?Generator/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:2*
dtype02A
?Generator/batch_normalization_1/FusedBatchNormV3/ReadVariableOpЇ
AGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:2*
dtype02C
AGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1│
0Generator/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_1/LeakyRelu:activations:06Generator/batch_normalization_1/ReadVariableOp:value:08Generator/batch_normalization_1/ReadVariableOp_1:value:0GGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         4,2:2:2:2:2:*
epsilon%oЃ:*
is_training( 22
0Generator/batch_normalization_1/FusedBatchNormV3ћ
Generator/covtr4/ShapeShape4Generator/batch_normalization_1/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr4/Shapeќ
$Generator/covtr4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr4/strided_slice/stackџ
&Generator/covtr4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr4/strided_slice/stack_1џ
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
Generator/covtr4/stack/2v
Generator/covtr4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :P2
Generator/covtr4/stack/3Э
Generator/covtr4/stackPack'Generator/covtr4/strided_slice:output:0!Generator/covtr4/stack/1:output:0!Generator/covtr4/stack/2:output:0!Generator/covtr4/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr4/stackџ
&Generator/covtr4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr4/strided_slice_1/stackъ
(Generator/covtr4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr4/strided_slice_1/stack_1ъ
(Generator/covtr4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr4/strided_slice_1/stack_2м
 Generator/covtr4/strided_slice_1StridedSliceGenerator/covtr4/stack:output:0/Generator/covtr4/strided_slice_1/stack:output:01Generator/covtr4/strided_slice_1/stack_1:output:01Generator/covtr4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr4/strided_slice_1Т
0Generator/covtr4/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:P2*
dtype022
0Generator/covtr4/conv2d_transpose/ReadVariableOpл
!Generator/covtr4/conv2d_transposeConv2DBackpropInputGenerator/covtr4/stack:output:08Generator/covtr4/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         hXP*
paddingSAME*
strides
2#
!Generator/covtr4/conv2d_transpose┐
'Generator/covtr4/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr4_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02)
'Generator/covtr4/BiasAdd/ReadVariableOpо
Generator/covtr4/BiasAddBiasAdd*Generator/covtr4/conv2d_transpose:output:0/Generator/covtr4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         hXP2
Generator/covtr4/BiasAddД
!Generator/leaky_re_lu_2/LeakyRelu	LeakyRelu!Generator/covtr4/BiasAdd:output:0*/
_output_shapes
:         hXP2#
!Generator/leaky_re_lu_2/LeakyReluн
.Generator/batch_normalization_2/ReadVariableOpReadVariableOp7generator_batch_normalization_2_readvariableop_resource*
_output_shapes
:P*
dtype020
.Generator/batch_normalization_2/ReadVariableOp┌
0Generator/batch_normalization_2/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:P*
dtype022
0Generator/batch_normalization_2/ReadVariableOp_1Є
?Generator/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02A
?Generator/batch_normalization_2/FusedBatchNormV3/ReadVariableOpЇ
AGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02C
AGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1│
0Generator/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_2/LeakyRelu:activations:06Generator/batch_normalization_2/ReadVariableOp:value:08Generator/batch_normalization_2/ReadVariableOp_1:value:0GGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         hXP:P:P:P:P:*
epsilon%oЃ:*
is_training( 22
0Generator/batch_normalization_2/FusedBatchNormV3љ
Generator/cov3/ShapeShape4Generator/batch_normalization_2/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/cov3/Shapeњ
"Generator/cov3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"Generator/cov3/strided_slice/stackќ
$Generator/cov3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$Generator/cov3/strided_slice/stack_1ќ
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
Generator/cov3/stack/3В
Generator/cov3/stackPack%Generator/cov3/strided_slice:output:0Generator/cov3/stack/1:output:0Generator/cov3/stack/2:output:0Generator/cov3/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/cov3/stackќ
$Generator/cov3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/cov3/strided_slice_1/stackџ
&Generator/cov3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/cov3/strided_slice_1/stack_1џ
&Generator/cov3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/cov3/strided_slice_1/stack_2к
Generator/cov3/strided_slice_1StridedSliceGenerator/cov3/stack:output:0-Generator/cov3/strided_slice_1/stack:output:0/Generator/cov3/strided_slice_1/stack_1:output:0/Generator/cov3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Generator/cov3/strided_slice_1Я
.Generator/cov3/conv2d_transpose/ReadVariableOpReadVariableOp7generator_cov3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:P*
dtype020
.Generator/cov3/conv2d_transpose/ReadVariableOp╔
Generator/cov3/conv2d_transposeConv2DBackpropInputGenerator/cov3/stack:output:06Generator/cov3/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_2/FusedBatchNormV3:y:0*
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
Generator/cov3/BiasAddќ
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
╚
Г
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_27767

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбAssignNewValueбAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           P:P:P:P:P:*
epsilon%oЃ:*
exponential_avg_factor%═╠L>2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЇ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1д
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           P2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           P::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           P
 
_user_specified_nameinputs
б
е
5__inference_batch_normalization_1_layer_call_fn_28983

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           2*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_276502
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           22

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           2::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           2
 
_user_specified_nameinputs
ў
Ѕ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_29031

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           P:P:P:P:P:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ѓ
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           P2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           P:::::i e
A
_output_shapes/
-:+                           P
 
_user_specified_nameinputs
ф
d
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_28988

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           P2
	LeakyReluЁ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           P2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           P:i e
A
_output_shapes/
-:+                           P
 
_user_specified_nameinputs
ё
G
+__inference_leaky_re_lu_layer_call_fn_28845

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
GPU2 *0J 8ѓ *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_279212
PartitionedCallє
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
 
_user_specified_nameinputs"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЪ
?
	gen_noise2
serving_default_gen_noise:0         d@
cov38
StatefulPartitionedCall:0         mYtensorflow/serving/predict:іћ
Эq
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
regularization_losses
	variables
trainable_variables
	keras_api

signatures
Ю__call__
+ъ&call_and_return_all_conditional_losses
Ъ_default_save_signature"йm
_tf_keras_networkАm{"class_name": "Functional", "name": "Generator", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Generator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gen_noise"}, "name": "gen_noise", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 429, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["gen_noise", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [13, 11, 3]}}, "name": "reshape", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr2", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu", "inbound_nodes": [[["covtr2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr3", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr3", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_1", "inbound_nodes": [[["covtr3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr4", "trainable": true, "dtype": "float32", "filters": 80, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr4", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_2", "inbound_nodes": [[["covtr4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "cov3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "cov3", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}], "input_layers": [["gen_noise", 0, 0]], "output_layers": [["cov3", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Generator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gen_noise"}, "name": "gen_noise", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 429, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["gen_noise", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [13, 11, 3]}}, "name": "reshape", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr2", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu", "inbound_nodes": [[["covtr2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr3", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr3", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_1", "inbound_nodes": [[["covtr3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr4", "trainable": true, "dtype": "float32", "filters": 80, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr4", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_2", "inbound_nodes": [[["covtr4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "cov3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "cov3", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}], "input_layers": [["gen_noise", 0, 0]], "output_layers": [["cov3", 0, 0]]}}}
ы"Ь
_tf_keras_input_layer╬{"class_name": "InputLayer", "name": "gen_noise", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gen_noise"}}
ы

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+а&call_and_return_all_conditional_losses
А__call__"╩
_tf_keras_layer░{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 429, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
э
	variables
regularization_losses
trainable_variables
	keras_api
+б&call_and_return_all_conditional_losses
Б__call__"Т
_tf_keras_layer╠{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [13, 11, 3]}}}
Г


kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
+ц&call_and_return_all_conditional_losses
Ц__call__"є	
_tf_keras_layerВ{"class_name": "Conv2DTranspose", "name": "covtr2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 11, 3]}}
▄
#	variables
$regularization_losses
%trainable_variables
&	keras_api
+д&call_and_return_all_conditional_losses
Д__call__"╦
_tf_keras_layer▒{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
и	
'axis
	(gamma
)beta
*moving_mean
+moving_variance
,	variables
-regularization_losses
.trainable_variables
/	keras_api
+е&call_and_return_all_conditional_losses
Е__call__"р
_tf_keras_layerК{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 22, 32]}}
»


0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
+ф&call_and_return_all_conditional_losses
Ф__call__"ѕ	
_tf_keras_layerЬ{"class_name": "Conv2DTranspose", "name": "covtr3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr3", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 22, 32]}}
Я
6	variables
7regularization_losses
8trainable_variables
9	keras_api
+г&call_and_return_all_conditional_losses
Г__call__"¤
_tf_keras_layerх{"class_name": "LeakyReLU", "name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
╗	
:axis
	;gamma
<beta
=moving_mean
>moving_variance
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
+«&call_and_return_all_conditional_losses
»__call__"т
_tf_keras_layer╦{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 52, 44, 50]}}
»


Ckernel
Dbias
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
+░&call_and_return_all_conditional_losses
▒__call__"ѕ	
_tf_keras_layerЬ{"class_name": "Conv2DTranspose", "name": "covtr4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr4", "trainable": true, "dtype": "float32", "filters": 80, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 52, 44, 50]}}
Я
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
+▓&call_and_return_all_conditional_losses
│__call__"¤
_tf_keras_layerх{"class_name": "LeakyReLU", "name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
╝	
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
+┤&call_and_return_all_conditional_losses
х__call__"Т
_tf_keras_layer╠{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 80}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 104, 88, 80]}}
Г


Vkernel
Wbias
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
+Х&call_and_return_all_conditional_losses
и__call__"є	
_tf_keras_layerВ{"class_name": "Conv2DTranspose", "name": "cov3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "cov3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 80}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 104, 88, 80]}}
 "
trackable_list_wrapper
к
0
1
2
3
(4
)5
*6
+7
08
19
;10
<11
=12
>13
C14
D15
N16
O17
P18
Q19
V20
W21"
trackable_list_wrapper
ќ
0
1
2
3
(4
)5
06
17
;8
<9
C10
D11
N12
O13
V14
W15"
trackable_list_wrapper
╬
\non_trainable_variables

]layers
^layer_regularization_losses
_metrics
regularization_losses
	variables
`layer_metrics
trainable_variables
Ю__call__
Ъ_default_save_signature
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
-
Иserving_default"
signature_map
:	dГ2dense/kernel
:Г2
dense/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
anon_trainable_variables
	variables
blayer_regularization_losses
cmetrics
regularization_losses

dlayers
elayer_metrics
trainable_variables
А__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
fnon_trainable_variables
	variables
glayer_regularization_losses
hmetrics
regularization_losses

ilayers
jlayer_metrics
trainable_variables
Б__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
':% 2covtr2/kernel
: 2covtr2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
knon_trainable_variables
	variables
llayer_regularization_losses
mmetrics
 regularization_losses

nlayers
olayer_metrics
!trainable_variables
Ц__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
pnon_trainable_variables
#	variables
qlayer_regularization_losses
rmetrics
$regularization_losses

slayers
tlayer_metrics
%trainable_variables
Д__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':% 2batch_normalization/gamma
&:$ 2batch_normalization/beta
/:-  (2batch_normalization/moving_mean
3:1  (2#batch_normalization/moving_variance
<
(0
)1
*2
+3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
░
unon_trainable_variables
,	variables
vlayer_regularization_losses
wmetrics
-regularization_losses

xlayers
ylayer_metrics
.trainable_variables
Е__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
':%2 2covtr3/kernel
:22covtr3/bias
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
░
znon_trainable_variables
2	variables
{layer_regularization_losses
|metrics
3regularization_losses

}layers
~layer_metrics
4trainable_variables
Ф__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
┤
non_trainable_variables
6	variables
 ђlayer_regularization_losses
Ђmetrics
7regularization_losses
ѓlayers
Ѓlayer_metrics
8trainable_variables
Г__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'22batch_normalization_1/gamma
(:&22batch_normalization_1/beta
1:/2 (2!batch_normalization_1/moving_mean
5:32 (2%batch_normalization_1/moving_variance
<
;0
<1
=2
>3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
х
ёnon_trainable_variables
?	variables
 Ёlayer_regularization_losses
єmetrics
@regularization_losses
Єlayers
ѕlayer_metrics
Atrainable_variables
»__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
':%P22covtr4/kernel
:P2covtr4/bias
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
х
Ѕnon_trainable_variables
E	variables
 іlayer_regularization_losses
Іmetrics
Fregularization_losses
їlayers
Їlayer_metrics
Gtrainable_variables
▒__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
јnon_trainable_variables
I	variables
 Јlayer_regularization_losses
љmetrics
Jregularization_losses
Љlayers
њlayer_metrics
Ktrainable_variables
│__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'P2batch_normalization_2/gamma
(:&P2batch_normalization_2/beta
1:/P (2!batch_normalization_2/moving_mean
5:3P (2%batch_normalization_2/moving_variance
<
N0
O1
P2
Q3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
х
Њnon_trainable_variables
R	variables
 ћlayer_regularization_losses
Ћmetrics
Sregularization_losses
ќlayers
Ќlayer_metrics
Ttrainable_variables
х__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
%:#P2cov3/kernel
:2	cov3/bias
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
х
ўnon_trainable_variables
X	variables
 Ўlayer_regularization_losses
џmetrics
Yregularization_losses
Џlayers
юlayer_metrics
Ztrainable_variables
и__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
J
*0
+1
=2
>3
P4
Q5"
trackable_list_wrapper
~
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
12"
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
*0
+1"
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
=0
>1"
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
P0
Q1"
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
Ы2№
)__inference_Generator_layer_call_fn_28246
)__inference_Generator_layer_call_fn_28796
)__inference_Generator_layer_call_fn_28747
)__inference_Generator_layer_call_fn_28355└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
я2█
D__inference_Generator_layer_call_and_return_conditional_losses_28136
D__inference_Generator_layer_call_and_return_conditional_losses_28698
D__inference_Generator_layer_call_and_return_conditional_losses_28555
D__inference_Generator_layer_call_and_return_conditional_losses_28076└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Я2П
 __inference__wrapped_model_27365И
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *(б%
#і 
	gen_noise         d
Ж2у
@__inference_dense_layer_call_and_return_conditional_losses_28807б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
¤2╠
%__inference_dense_layer_call_fn_28816б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_reshape_layer_call_and_return_conditional_losses_28830б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_reshape_layer_call_fn_28835б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
а2Ю
A__inference_covtr2_layer_call_and_return_conditional_losses_27399О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           
Ё2ѓ
&__inference_covtr2_layer_call_fn_27409О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           
­2ь
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_28840б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_leaky_re_lu_layer_call_fn_28845б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┌2О
N__inference_batch_normalization_layer_call_and_return_conditional_losses_28883
N__inference_batch_normalization_layer_call_and_return_conditional_losses_28865┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ц2А
3__inference_batch_normalization_layer_call_fn_28909
3__inference_batch_normalization_layer_call_fn_28896┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
а2Ю
A__inference_covtr3_layer_call_and_return_conditional_losses_27547О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
Ё2ѓ
&__inference_covtr3_layer_call_fn_27557О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
Ы2№
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_28914б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_leaky_re_lu_1_layer_call_fn_28919б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
я2█
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_28957
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_28939┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Ц
5__inference_batch_normalization_1_layer_call_fn_28983
5__inference_batch_normalization_1_layer_call_fn_28970┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
а2Ю
A__inference_covtr4_layer_call_and_return_conditional_losses_27695О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           2
Ё2ѓ
&__inference_covtr4_layer_call_fn_27705О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           2
Ы2№
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_28988б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_leaky_re_lu_2_layer_call_fn_28993б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
я2█
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_29031
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_29013┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Ц
5__inference_batch_normalization_2_layer_call_fn_29057
5__inference_batch_normalization_2_layer_call_fn_29044┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ъ2Џ
?__inference_cov3_layer_call_and_return_conditional_losses_27848О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           P
Ѓ2ђ
$__inference_cov3_layer_call_fn_27858О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           P
4B2
#__inference_signature_wrapper_28406	gen_noiseя
D__inference_Generator_layer_call_and_return_conditional_losses_28076Ћ()*+01;<=>CDNOPQVW:б7
0б-
#і 
	gen_noise         d
p

 
ф "?б<
5і2
0+                           
џ я
D__inference_Generator_layer_call_and_return_conditional_losses_28136Ћ()*+01;<=>CDNOPQVW:б7
0б-
#і 
	gen_noise         d
p 

 
ф "?б<
5і2
0+                           
џ ╔
D__inference_Generator_layer_call_and_return_conditional_losses_28555ђ()*+01;<=>CDNOPQVW7б4
-б*
 і
inputs         d
p

 
ф "-б*
#і 
0         mY
џ ╔
D__inference_Generator_layer_call_and_return_conditional_losses_28698ђ()*+01;<=>CDNOPQVW7б4
-б*
 і
inputs         d
p 

 
ф "-б*
#і 
0         mY
џ Х
)__inference_Generator_layer_call_fn_28246ѕ()*+01;<=>CDNOPQVW:б7
0б-
#і 
	gen_noise         d
p

 
ф "2і/+                           Х
)__inference_Generator_layer_call_fn_28355ѕ()*+01;<=>CDNOPQVW:б7
0б-
#і 
	gen_noise         d
p 

 
ф "2і/+                           │
)__inference_Generator_layer_call_fn_28747Ё()*+01;<=>CDNOPQVW7б4
-б*
 і
inputs         d
p

 
ф "2і/+                           │
)__inference_Generator_layer_call_fn_28796Ё()*+01;<=>CDNOPQVW7б4
-б*
 і
inputs         d
p 

 
ф "2і/+                           д
 __inference__wrapped_model_27365Ђ()*+01;<=>CDNOPQVW2б/
(б%
#і 
	gen_noise         d
ф "3ф0
.
cov3&і#
cov3         mYв
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_28939ќ;<=>MбJ
Cб@
:і7
inputs+                           2
p
ф "?б<
5і2
0+                           2
џ в
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_28957ќ;<=>MбJ
Cб@
:і7
inputs+                           2
p 
ф "?б<
5і2
0+                           2
џ ├
5__inference_batch_normalization_1_layer_call_fn_28970Ѕ;<=>MбJ
Cб@
:і7
inputs+                           2
p
ф "2і/+                           2├
5__inference_batch_normalization_1_layer_call_fn_28983Ѕ;<=>MбJ
Cб@
:і7
inputs+                           2
p 
ф "2і/+                           2в
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_29013ќNOPQMбJ
Cб@
:і7
inputs+                           P
p
ф "?б<
5і2
0+                           P
џ в
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_29031ќNOPQMбJ
Cб@
:і7
inputs+                           P
p 
ф "?б<
5і2
0+                           P
џ ├
5__inference_batch_normalization_2_layer_call_fn_29044ЅNOPQMбJ
Cб@
:і7
inputs+                           P
p
ф "2і/+                           P├
5__inference_batch_normalization_2_layer_call_fn_29057ЅNOPQMбJ
Cб@
:і7
inputs+                           P
p 
ф "2і/+                           Pж
N__inference_batch_normalization_layer_call_and_return_conditional_losses_28865ќ()*+MбJ
Cб@
:і7
inputs+                            
p
ф "?б<
5і2
0+                            
џ ж
N__inference_batch_normalization_layer_call_and_return_conditional_losses_28883ќ()*+MбJ
Cб@
:і7
inputs+                            
p 
ф "?б<
5і2
0+                            
џ ┴
3__inference_batch_normalization_layer_call_fn_28896Ѕ()*+MбJ
Cб@
:і7
inputs+                            
p
ф "2і/+                            ┴
3__inference_batch_normalization_layer_call_fn_28909Ѕ()*+MбJ
Cб@
:і7
inputs+                            
p 
ф "2і/+                            н
?__inference_cov3_layer_call_and_return_conditional_losses_27848љVWIбF
?б<
:і7
inputs+                           P
ф "?б<
5і2
0+                           
џ г
$__inference_cov3_layer_call_fn_27858ЃVWIбF
?б<
:і7
inputs+                           P
ф "2і/+                           о
A__inference_covtr2_layer_call_and_return_conditional_losses_27399љIбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                            
џ «
&__inference_covtr2_layer_call_fn_27409ЃIбF
?б<
:і7
inputs+                           
ф "2і/+                            о
A__inference_covtr3_layer_call_and_return_conditional_losses_27547љ01IбF
?б<
:і7
inputs+                            
ф "?б<
5і2
0+                           2
џ «
&__inference_covtr3_layer_call_fn_27557Ѓ01IбF
?б<
:і7
inputs+                            
ф "2і/+                           2о
A__inference_covtr4_layer_call_and_return_conditional_losses_27695љCDIбF
?б<
:і7
inputs+                           2
ф "?б<
5і2
0+                           P
џ «
&__inference_covtr4_layer_call_fn_27705ЃCDIбF
?б<
:і7
inputs+                           2
ф "2і/+                           PА
@__inference_dense_layer_call_and_return_conditional_losses_28807]/б,
%б"
 і
inputs         d
ф "&б#
і
0         Г
џ y
%__inference_dense_layer_call_fn_28816P/б,
%б"
 і
inputs         d
ф "і         Г┘
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_28914їIбF
?б<
:і7
inputs+                           2
ф "?б<
5і2
0+                           2
џ ░
-__inference_leaky_re_lu_1_layer_call_fn_28919IбF
?б<
:і7
inputs+                           2
ф "2і/+                           2┘
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_28988їIбF
?б<
:і7
inputs+                           P
ф "?б<
5і2
0+                           P
џ ░
-__inference_leaky_re_lu_2_layer_call_fn_28993IбF
?б<
:і7
inputs+                           P
ф "2і/+                           PО
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_28840їIбF
?б<
:і7
inputs+                            
ф "?б<
5і2
0+                            
џ «
+__inference_leaky_re_lu_layer_call_fn_28845IбF
?б<
:і7
inputs+                            
ф "2і/+                            Д
B__inference_reshape_layer_call_and_return_conditional_losses_28830a0б-
&б#
!і
inputs         Г
ф "-б*
#і 
0         
џ 
'__inference_reshape_layer_call_fn_28835T0б-
&б#
!і
inputs         Г
ф " і         Х
#__inference_signature_wrapper_28406ј()*+01;<=>CDNOPQVW?б<
б 
5ф2
0
	gen_noise#і 
	gen_noise         d"3ф0
.
cov3&і#
cov3         mY