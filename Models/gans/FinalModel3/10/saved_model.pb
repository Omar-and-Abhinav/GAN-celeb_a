��
��
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
dtypetype�
�
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
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.12v2.3.0-54-gfcc4b966f18��
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	d�*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
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
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
: *
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance
�
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
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:2*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:2*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:2*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*6
shared_name'%batch_normalization_1/moving_variance
�
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
�
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*,
shared_namebatch_normalization_2/gamma
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:P*
dtype0
�
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_namebatch_normalization_2/beta
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:P*
dtype0
�
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*2
shared_name#!batch_normalization_2/moving_mean
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:P*
dtype0
�
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*6
shared_name'%batch_normalization_2/moving_variance
�
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
�9
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�9
value�9B�8 B�8
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
non_trainable_variables
6	variables
 �layer_regularization_losses
�metrics
7regularization_losses
�layers
�layer_metrics
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
�
�non_trainable_variables
?	variables
 �layer_regularization_losses
�metrics
@regularization_losses
�layers
�layer_metrics
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
�
�non_trainable_variables
E	variables
 �layer_regularization_losses
�metrics
Fregularization_losses
�layers
�layer_metrics
Gtrainable_variables
 
 
 
�
�non_trainable_variables
I	variables
 �layer_regularization_losses
�metrics
Jregularization_losses
�layers
�layer_metrics
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
�
�non_trainable_variables
R	variables
 �layer_regularization_losses
�metrics
Sregularization_losses
�layers
�layer_metrics
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
�
�non_trainable_variables
X	variables
 �layer_regularization_losses
�metrics
Yregularization_losses
�layers
�layer_metrics
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
:���������d*
dtype0*
shape:���������d
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_gen_noisedense/kernel
dense/biascovtr2/kernelcovtr2/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancecovtr3/kernelcovtr3/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancecovtr4/kernelcovtr4/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancecov3/kernel	cov3/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������mY*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *-
f(R&
$__inference_signature_wrapper_142175
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
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
GPU2 *0J 8� *(
f#R!
__inference__traced_save_142915
�
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
GPU2 *0J 8� *+
f&R$
"__inference__traced_restore_142991��
�>
�
E__inference_Generator_layer_call_and_return_conditional_losses_141905
	gen_noise
dense_141848
dense_141850
covtr2_141854
covtr2_141856
batch_normalization_141860
batch_normalization_141862
batch_normalization_141864
batch_normalization_141866
covtr3_141869
covtr3_141871 
batch_normalization_1_141875 
batch_normalization_1_141877 
batch_normalization_1_141879 
batch_normalization_1_141881
covtr4_141884
covtr4_141886 
batch_normalization_2_141890 
batch_normalization_2_141892 
batch_normalization_2_141894 
batch_normalization_2_141896
cov3_141899
cov3_141901
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�cov3/StatefulPartitionedCall�covtr2/StatefulPartitionedCall�covtr3/StatefulPartitionedCall�covtr4/StatefulPartitionedCall�dense/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall	gen_noisedense_141848dense_141850*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1416422
dense/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1416722
reshape/PartitionedCall�
covtr2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr2_141854covtr2_141856*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr2_layer_call_and_return_conditional_losses_1411682 
covtr2/StatefulPartitionedCall�
leaky_re_lu/PartitionedCallPartitionedCall'covtr2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_1416902
leaky_re_lu/PartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_141860batch_normalization_141862batch_normalization_141864batch_normalization_141866*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1412712-
+batch_normalization/StatefulPartitionedCall�
covtr3/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr3_141869covtr3_141871*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr3_layer_call_and_return_conditional_losses_1413162 
covtr3/StatefulPartitionedCall�
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_1417432
leaky_re_lu_1/PartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_141875batch_normalization_1_141877batch_normalization_1_141879batch_normalization_1_141881*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������2*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1414192/
-batch_normalization_1/StatefulPartitionedCall�
covtr4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr4_141884covtr4_141886*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr4_layer_call_and_return_conditional_losses_1414642 
covtr4/StatefulPartitionedCall�
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_1417962
leaky_re_lu_2/PartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_141890batch_normalization_2_141892batch_normalization_2_141894batch_normalization_2_141896*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������P*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1415672/
-batch_normalization_2/StatefulPartitionedCall�
cov3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0cov3_141899cov3_141901*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_cov3_layer_call_and_return_conditional_losses_1416172
cov3/StatefulPartitionedCall�
IdentityIdentity%cov3/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^cov3/StatefulPartitionedCall^covtr2/StatefulPartitionedCall^covtr3/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::2Z
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
:���������d
#
_user_specified_name	gen_noise
�
�
4__inference_batch_normalization_layer_call_fn_142665

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1412402
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�!
�
B__inference_covtr2_layer_call_and_return_conditional_losses_141168

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity�D
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
strided_slice/stack_2�
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
strided_slice_1/stack_2�
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
strided_slice_2/stack_2�
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
stack/3�
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
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�>
�
E__inference_Generator_layer_call_and_return_conditional_losses_142077

inputs
dense_142020
dense_142022
covtr2_142026
covtr2_142028
batch_normalization_142032
batch_normalization_142034
batch_normalization_142036
batch_normalization_142038
covtr3_142041
covtr3_142043 
batch_normalization_1_142047 
batch_normalization_1_142049 
batch_normalization_1_142051 
batch_normalization_1_142053
covtr4_142056
covtr4_142058 
batch_normalization_2_142062 
batch_normalization_2_142064 
batch_normalization_2_142066 
batch_normalization_2_142068
cov3_142071
cov3_142073
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�cov3/StatefulPartitionedCall�covtr2/StatefulPartitionedCall�covtr3/StatefulPartitionedCall�covtr4/StatefulPartitionedCall�dense/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_142020dense_142022*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1416422
dense/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1416722
reshape/PartitionedCall�
covtr2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr2_142026covtr2_142028*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr2_layer_call_and_return_conditional_losses_1411682 
covtr2/StatefulPartitionedCall�
leaky_re_lu/PartitionedCallPartitionedCall'covtr2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_1416902
leaky_re_lu/PartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_142032batch_normalization_142034batch_normalization_142036batch_normalization_142038*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1412712-
+batch_normalization/StatefulPartitionedCall�
covtr3/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr3_142041covtr3_142043*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr3_layer_call_and_return_conditional_losses_1413162 
covtr3/StatefulPartitionedCall�
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_1417432
leaky_re_lu_1/PartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_142047batch_normalization_1_142049batch_normalization_1_142051batch_normalization_1_142053*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������2*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1414192/
-batch_normalization_1/StatefulPartitionedCall�
covtr4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr4_142056covtr4_142058*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr4_layer_call_and_return_conditional_losses_1414642 
covtr4/StatefulPartitionedCall�
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_1417962
leaky_re_lu_2/PartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_142062batch_normalization_2_142064batch_normalization_2_142066batch_normalization_2_142068*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������P*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1415672/
-batch_normalization_2/StatefulPartitionedCall�
cov3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0cov3_142071cov3_142073*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_cov3_layer_call_and_return_conditional_losses_1416172
cov3/StatefulPartitionedCall�
IdentityIdentity%cov3/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^cov3/StatefulPartitionedCall^covtr2/StatefulPartitionedCall^covtr3/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::2Z
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
:���������d
 
_user_specified_nameinputs
�
�
A__inference_dense_layer_call_and_return_conditional_losses_142576

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_141567

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������P:P:P:P:P:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������P2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������P:::::i e
A
_output_shapes/
-:+���������������������������P
 
_user_specified_nameinputs
�
�
*__inference_Generator_layer_call_fn_142516

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
identity��StatefulPartitionedCall�
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
-:+���������������������������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_Generator_layer_call_and_return_conditional_losses_1419682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
_
C__inference_reshape_layer_call_and_return_conditional_losses_142599

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
strided_slice/stack_2�
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
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�7
�

__inference__traced_save_142915
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

identity_1��MergeV2Checkpoints�
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
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_2c96c0b1cfc34cfd8d1d18010da756c1/part2	
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop(savev2_covtr2_kernel_read_readvariableop&savev2_covtr2_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop(savev2_covtr3_kernel_read_readvariableop&savev2_covtr3_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop(savev2_covtr4_kernel_read_readvariableop&savev2_covtr4_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop&savev2_cov3_kernel_read_readvariableop$savev2_cov3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :	d�:�: : : : : : :2 :2:2:2:2:2:P2:P:P:P:P:P:P:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	d�:!

_output_shapes	
:�:,(
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
�
�
$__inference_signature_wrapper_142175
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
identity��StatefulPartitionedCall�
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
:���������mY*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� **
f%R#
!__inference__wrapped_model_1411342
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������mY2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:���������d
#
_user_specified_name	gen_noise
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_141240

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%��L>2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_2_layer_call_fn_142813

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1415362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������P2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������P::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������P
 
_user_specified_nameinputs
�
J
.__inference_leaky_re_lu_2_layer_call_fn_142762

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_1417962
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+���������������������������P2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������P:i e
A
_output_shapes/
-:+���������������������������P
 
_user_specified_nameinputs
��
�

E__inference_Generator_layer_call_and_return_conditional_losses_142467

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
identity��
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2

dense/Reluf
reshape/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape�
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack�
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1�
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2�
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
reshape/Reshape/shape/3�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape�
reshape/ReshapeReshapedense/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2
reshape/Reshaped
covtr2/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
covtr2/Shape�
covtr2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr2/strided_slice/stack�
covtr2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr2/strided_slice/stack_1�
covtr2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr2/strided_slice/stack_2�
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
covtr2/stack/3�
covtr2/stackPackcovtr2/strided_slice:output:0covtr2/stack/1:output:0covtr2/stack/2:output:0covtr2/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr2/stack�
covtr2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr2/strided_slice_1/stack�
covtr2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr2/strided_slice_1/stack_1�
covtr2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr2/strided_slice_1/stack_2�
covtr2/strided_slice_1StridedSlicecovtr2/stack:output:0%covtr2/strided_slice_1/stack:output:0'covtr2/strided_slice_1/stack_1:output:0'covtr2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr2/strided_slice_1�
&covtr2/conv2d_transpose/ReadVariableOpReadVariableOp/covtr2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02(
&covtr2/conv2d_transpose/ReadVariableOp�
covtr2/conv2d_transposeConv2DBackpropInputcovtr2/stack:output:0.covtr2/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
covtr2/conv2d_transpose�
covtr2/BiasAdd/ReadVariableOpReadVariableOp&covtr2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
covtr2/BiasAdd/ReadVariableOp�
covtr2/BiasAddBiasAdd covtr2/conv2d_transpose:output:0%covtr2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
covtr2/BiasAdd�
leaky_re_lu/LeakyRelu	LeakyRelucovtr2/BiasAdd:output:0*/
_output_shapes
:��������� 2
leaky_re_lu/LeakyRelu�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3#leaky_re_lu/LeakyRelu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( 2&
$batch_normalization/FusedBatchNormV3t
covtr3/ShapeShape(batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr3/Shape�
covtr3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr3/strided_slice/stack�
covtr3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr3/strided_slice/stack_1�
covtr3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr3/strided_slice/stack_2�
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
covtr3/stack/3�
covtr3/stackPackcovtr3/strided_slice:output:0covtr3/stack/1:output:0covtr3/stack/2:output:0covtr3/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr3/stack�
covtr3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr3/strided_slice_1/stack�
covtr3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr3/strided_slice_1/stack_1�
covtr3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr3/strided_slice_1/stack_2�
covtr3/strided_slice_1StridedSlicecovtr3/stack:output:0%covtr3/strided_slice_1/stack:output:0'covtr3/strided_slice_1/stack_1:output:0'covtr3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr3/strided_slice_1�
&covtr3/conv2d_transpose/ReadVariableOpReadVariableOp/covtr3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:2 *
dtype02(
&covtr3/conv2d_transpose/ReadVariableOp�
covtr3/conv2d_transposeConv2DBackpropInputcovtr3/stack:output:0.covtr3/conv2d_transpose/ReadVariableOp:value:0(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������4,2*
paddingSAME*
strides
2
covtr3/conv2d_transpose�
covtr3/BiasAdd/ReadVariableOpReadVariableOp&covtr3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
covtr3/BiasAdd/ReadVariableOp�
covtr3/BiasAddBiasAdd covtr3/conv2d_transpose:output:0%covtr3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������4,22
covtr3/BiasAdd�
leaky_re_lu_1/LeakyRelu	LeakyRelucovtr3/BiasAdd:output:0*/
_output_shapes
:���������4,22
leaky_re_lu_1/LeakyRelu�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:2*
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:2*
dtype02(
&batch_normalization_1/ReadVariableOp_1�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:2*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:2*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_1/LeakyRelu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������4,2:2:2:2:2:*
epsilon%o�:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3v
covtr4/ShapeShape*batch_normalization_1/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr4/Shape�
covtr4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr4/strided_slice/stack�
covtr4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr4/strided_slice/stack_1�
covtr4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr4/strided_slice/stack_2�
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
covtr4/stack/3�
covtr4/stackPackcovtr4/strided_slice:output:0covtr4/stack/1:output:0covtr4/stack/2:output:0covtr4/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr4/stack�
covtr4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr4/strided_slice_1/stack�
covtr4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr4/strided_slice_1/stack_1�
covtr4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr4/strided_slice_1/stack_2�
covtr4/strided_slice_1StridedSlicecovtr4/stack:output:0%covtr4/strided_slice_1/stack:output:0'covtr4/strided_slice_1/stack_1:output:0'covtr4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr4/strided_slice_1�
&covtr4/conv2d_transpose/ReadVariableOpReadVariableOp/covtr4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:P2*
dtype02(
&covtr4/conv2d_transpose/ReadVariableOp�
covtr4/conv2d_transposeConv2DBackpropInputcovtr4/stack:output:0.covtr4/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������hXP*
paddingSAME*
strides
2
covtr4/conv2d_transpose�
covtr4/BiasAdd/ReadVariableOpReadVariableOp&covtr4_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
covtr4/BiasAdd/ReadVariableOp�
covtr4/BiasAddBiasAdd covtr4/conv2d_transpose:output:0%covtr4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������hXP2
covtr4/BiasAdd�
leaky_re_lu_2/LeakyRelu	LeakyRelucovtr4/BiasAdd:output:0*/
_output_shapes
:���������hXP2
leaky_re_lu_2/LeakyRelu�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:P*
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:P*
dtype02(
&batch_normalization_2/ReadVariableOp_1�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_2/LeakyRelu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������hXP:P:P:P:P:*
epsilon%o�:*
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
cov3/strided_slice/stack�
cov3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice/stack_1�
cov3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice/stack_2�
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
cov3/stack/3�

cov3/stackPackcov3/strided_slice:output:0cov3/stack/1:output:0cov3/stack/2:output:0cov3/stack/3:output:0*
N*
T0*
_output_shapes
:2

cov3/stack�
cov3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cov3/strided_slice_1/stack�
cov3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice_1/stack_1�
cov3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice_1/stack_2�
cov3/strided_slice_1StridedSlicecov3/stack:output:0#cov3/strided_slice_1/stack:output:0%cov3/strided_slice_1/stack_1:output:0%cov3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cov3/strided_slice_1�
$cov3/conv2d_transpose/ReadVariableOpReadVariableOp-cov3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:P*
dtype02&
$cov3/conv2d_transpose/ReadVariableOp�
cov3/conv2d_transposeConv2DBackpropInputcov3/stack:output:0,cov3/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������mY*
paddingVALID*
strides
2
cov3/conv2d_transpose�
cov3/BiasAdd/ReadVariableOpReadVariableOp$cov3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
cov3/BiasAdd/ReadVariableOp�
cov3/BiasAddBiasAddcov3/conv2d_transpose:output:0#cov3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������mY2
cov3/BiasAddx
cov3/SigmoidSigmoidcov3/BiasAdd:output:0*
T0*/
_output_shapes
:���������mY2
cov3/Sigmoidl
IdentityIdentitycov3/Sigmoid:y:0*
T0*/
_output_shapes
:���������mY2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d:::::::::::::::::::::::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_141271

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� :::::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�!
�
B__inference_covtr4_layer_call_and_return_conditional_losses_141464

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity�D
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
strided_slice/stack_2�
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
strided_slice_1/stack_2�
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
strided_slice_2/stack_2�
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
stack/3�
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
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:P2*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������P*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������P2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������P2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������2:::i e
A
_output_shapes/
-:+���������������������������2
 
_user_specified_nameinputs
�_
�
"__inference__traced_restore_142991
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
identity_23��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
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

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp assignvariableop_2_covtr2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_covtr2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp,assignvariableop_4_batch_normalization_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp+assignvariableop_5_batch_normalization_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp2assignvariableop_6_batch_normalization_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp assignvariableop_8_covtr3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_covtr3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_1_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_1_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_1_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_1_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp!assignvariableop_14_covtr4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_covtr4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_2_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_2_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_2_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_2_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_cov3_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpassignvariableop_21_cov3_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_219
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22�
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
�
_
C__inference_reshape_layer_call_and_return_conditional_losses_141672

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
strided_slice/stack_2�
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
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_2_layer_call_fn_142826

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������P*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1415672
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������P2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������P::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������P
 
_user_specified_nameinputs
�
�
*__inference_Generator_layer_call_fn_142565

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
identity��StatefulPartitionedCall�
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
-:+���������������������������*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_Generator_layer_call_and_return_conditional_losses_1420772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
c
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_142609

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+��������������������������� 2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+��������������������������� :i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�>
�
E__inference_Generator_layer_call_and_return_conditional_losses_141968

inputs
dense_141911
dense_141913
covtr2_141917
covtr2_141919
batch_normalization_141923
batch_normalization_141925
batch_normalization_141927
batch_normalization_141929
covtr3_141932
covtr3_141934 
batch_normalization_1_141938 
batch_normalization_1_141940 
batch_normalization_1_141942 
batch_normalization_1_141944
covtr4_141947
covtr4_141949 
batch_normalization_2_141953 
batch_normalization_2_141955 
batch_normalization_2_141957 
batch_normalization_2_141959
cov3_141962
cov3_141964
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�cov3/StatefulPartitionedCall�covtr2/StatefulPartitionedCall�covtr3/StatefulPartitionedCall�covtr4/StatefulPartitionedCall�dense/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_141911dense_141913*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1416422
dense/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1416722
reshape/PartitionedCall�
covtr2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr2_141917covtr2_141919*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr2_layer_call_and_return_conditional_losses_1411682 
covtr2/StatefulPartitionedCall�
leaky_re_lu/PartitionedCallPartitionedCall'covtr2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_1416902
leaky_re_lu/PartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_141923batch_normalization_141925batch_normalization_141927batch_normalization_141929*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1412402-
+batch_normalization/StatefulPartitionedCall�
covtr3/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr3_141932covtr3_141934*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr3_layer_call_and_return_conditional_losses_1413162 
covtr3/StatefulPartitionedCall�
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_1417432
leaky_re_lu_1/PartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_141938batch_normalization_1_141940batch_normalization_1_141942batch_normalization_1_141944*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1413882/
-batch_normalization_1/StatefulPartitionedCall�
covtr4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr4_141947covtr4_141949*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr4_layer_call_and_return_conditional_losses_1414642 
covtr4/StatefulPartitionedCall�
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_1417962
leaky_re_lu_2/PartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_141953batch_normalization_2_141955batch_normalization_2_141957batch_normalization_2_141959*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1415362/
-batch_normalization_2/StatefulPartitionedCall�
cov3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0cov3_141962cov3_141964*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_cov3_layer_call_and_return_conditional_losses_1416172
cov3/StatefulPartitionedCall�
IdentityIdentity%cov3/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^cov3/StatefulPartitionedCall^covtr2/StatefulPartitionedCall^covtr3/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::2Z
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
:���������d
 
_user_specified_nameinputs
�>
�
E__inference_Generator_layer_call_and_return_conditional_losses_141845
	gen_noise
dense_141653
dense_141655
covtr2_141680
covtr2_141682
batch_normalization_141724
batch_normalization_141726
batch_normalization_141728
batch_normalization_141730
covtr3_141733
covtr3_141735 
batch_normalization_1_141777 
batch_normalization_1_141779 
batch_normalization_1_141781 
batch_normalization_1_141783
covtr4_141786
covtr4_141788 
batch_normalization_2_141830 
batch_normalization_2_141832 
batch_normalization_2_141834 
batch_normalization_2_141836
cov3_141839
cov3_141841
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�cov3/StatefulPartitionedCall�covtr2/StatefulPartitionedCall�covtr3/StatefulPartitionedCall�covtr4/StatefulPartitionedCall�dense/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall	gen_noisedense_141653dense_141655*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1416422
dense/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1416722
reshape/PartitionedCall�
covtr2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr2_141680covtr2_141682*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr2_layer_call_and_return_conditional_losses_1411682 
covtr2/StatefulPartitionedCall�
leaky_re_lu/PartitionedCallPartitionedCall'covtr2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_1416902
leaky_re_lu/PartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_141724batch_normalization_141726batch_normalization_141728batch_normalization_141730*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1412402-
+batch_normalization/StatefulPartitionedCall�
covtr3/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr3_141733covtr3_141735*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr3_layer_call_and_return_conditional_losses_1413162 
covtr3/StatefulPartitionedCall�
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_1417432
leaky_re_lu_1/PartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_141777batch_normalization_1_141779batch_normalization_1_141781batch_normalization_1_141783*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1413882/
-batch_normalization_1/StatefulPartitionedCall�
covtr4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr4_141786covtr4_141788*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr4_layer_call_and_return_conditional_losses_1414642 
covtr4/StatefulPartitionedCall�
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_1417962
leaky_re_lu_2/PartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_141830batch_normalization_2_141832batch_normalization_2_141834batch_normalization_2_141836*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1415362/
-batch_normalization_2/StatefulPartitionedCall�
cov3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0cov3_141839cov3_141841*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_cov3_layer_call_and_return_conditional_losses_1416172
cov3/StatefulPartitionedCall�
IdentityIdentity%cov3/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^cov3/StatefulPartitionedCall^covtr2/StatefulPartitionedCall^covtr3/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::2Z
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
:���������d
#
_user_specified_name	gen_noise
�
|
'__inference_covtr4_layer_call_fn_141474

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr4_layer_call_and_return_conditional_losses_1414642
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������P2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������2::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������2
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_142652

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� :::::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_1_layer_call_fn_142739

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1413882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������22

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������2::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������2
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_layer_call_fn_142678

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1412712
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_142708

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:2*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:2*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:2*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:2*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������2:2:2:2:2:*
epsilon%o�:*
exponential_avg_factor%��L>2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+���������������������������22

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������2::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������2
 
_user_specified_nameinputs
�
�
!__inference__wrapped_model_141134
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
identity��
%Generator/dense/MatMul/ReadVariableOpReadVariableOp.generator_dense_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype02'
%Generator/dense/MatMul/ReadVariableOp�
Generator/dense/MatMulMatMul	gen_noise-Generator/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Generator/dense/MatMul�
&Generator/dense/BiasAdd/ReadVariableOpReadVariableOp/generator_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&Generator/dense/BiasAdd/ReadVariableOp�
Generator/dense/BiasAddBiasAdd Generator/dense/MatMul:product:0.Generator/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Generator/dense/BiasAdd�
Generator/dense/ReluRelu Generator/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
Generator/dense/Relu�
Generator/reshape/ShapeShape"Generator/dense/Relu:activations:0*
T0*
_output_shapes
:2
Generator/reshape/Shape�
%Generator/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Generator/reshape/strided_slice/stack�
'Generator/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/reshape/strided_slice/stack_1�
'Generator/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/reshape/strided_slice/stack_2�
Generator/reshape/strided_sliceStridedSlice Generator/reshape/Shape:output:0.Generator/reshape/strided_slice/stack:output:00Generator/reshape/strided_slice/stack_1:output:00Generator/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
Generator/reshape/strided_slice�
!Generator/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!Generator/reshape/Reshape/shape/1�
!Generator/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!Generator/reshape/Reshape/shape/2�
!Generator/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!Generator/reshape/Reshape/shape/3�
Generator/reshape/Reshape/shapePack(Generator/reshape/strided_slice:output:0*Generator/reshape/Reshape/shape/1:output:0*Generator/reshape/Reshape/shape/2:output:0*Generator/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
Generator/reshape/Reshape/shape�
Generator/reshape/ReshapeReshape"Generator/dense/Relu:activations:0(Generator/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2
Generator/reshape/Reshape�
Generator/covtr2/ShapeShape"Generator/reshape/Reshape:output:0*
T0*
_output_shapes
:2
Generator/covtr2/Shape�
$Generator/covtr2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr2/strided_slice/stack�
&Generator/covtr2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr2/strided_slice/stack_1�
&Generator/covtr2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr2/strided_slice/stack_2�
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
Generator/covtr2/stack/3�
Generator/covtr2/stackPack'Generator/covtr2/strided_slice:output:0!Generator/covtr2/stack/1:output:0!Generator/covtr2/stack/2:output:0!Generator/covtr2/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr2/stack�
&Generator/covtr2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr2/strided_slice_1/stack�
(Generator/covtr2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr2/strided_slice_1/stack_1�
(Generator/covtr2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr2/strided_slice_1/stack_2�
 Generator/covtr2/strided_slice_1StridedSliceGenerator/covtr2/stack:output:0/Generator/covtr2/strided_slice_1/stack:output:01Generator/covtr2/strided_slice_1/stack_1:output:01Generator/covtr2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr2/strided_slice_1�
0Generator/covtr2/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype022
0Generator/covtr2/conv2d_transpose/ReadVariableOp�
!Generator/covtr2/conv2d_transposeConv2DBackpropInputGenerator/covtr2/stack:output:08Generator/covtr2/conv2d_transpose/ReadVariableOp:value:0"Generator/reshape/Reshape:output:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2#
!Generator/covtr2/conv2d_transpose�
'Generator/covtr2/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'Generator/covtr2/BiasAdd/ReadVariableOp�
Generator/covtr2/BiasAddBiasAdd*Generator/covtr2/conv2d_transpose:output:0/Generator/covtr2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
Generator/covtr2/BiasAdd�
Generator/leaky_re_lu/LeakyRelu	LeakyRelu!Generator/covtr2/BiasAdd:output:0*/
_output_shapes
:��������� 2!
Generator/leaky_re_lu/LeakyRelu�
,Generator/batch_normalization/ReadVariableOpReadVariableOp5generator_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02.
,Generator/batch_normalization/ReadVariableOp�
.Generator/batch_normalization/ReadVariableOp_1ReadVariableOp7generator_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype020
.Generator/batch_normalization/ReadVariableOp_1�
=Generator/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpFgenerator_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02?
=Generator/batch_normalization/FusedBatchNormV3/ReadVariableOp�
?Generator/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHgenerator_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02A
?Generator/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
.Generator/batch_normalization/FusedBatchNormV3FusedBatchNormV3-Generator/leaky_re_lu/LeakyRelu:activations:04Generator/batch_normalization/ReadVariableOp:value:06Generator/batch_normalization/ReadVariableOp_1:value:0EGenerator/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0GGenerator/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( 20
.Generator/batch_normalization/FusedBatchNormV3�
Generator/covtr3/ShapeShape2Generator/batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr3/Shape�
$Generator/covtr3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr3/strided_slice/stack�
&Generator/covtr3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr3/strided_slice/stack_1�
&Generator/covtr3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr3/strided_slice/stack_2�
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
Generator/covtr3/stack/3�
Generator/covtr3/stackPack'Generator/covtr3/strided_slice:output:0!Generator/covtr3/stack/1:output:0!Generator/covtr3/stack/2:output:0!Generator/covtr3/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr3/stack�
&Generator/covtr3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr3/strided_slice_1/stack�
(Generator/covtr3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr3/strided_slice_1/stack_1�
(Generator/covtr3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr3/strided_slice_1/stack_2�
 Generator/covtr3/strided_slice_1StridedSliceGenerator/covtr3/stack:output:0/Generator/covtr3/strided_slice_1/stack:output:01Generator/covtr3/strided_slice_1/stack_1:output:01Generator/covtr3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr3/strided_slice_1�
0Generator/covtr3/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:2 *
dtype022
0Generator/covtr3/conv2d_transpose/ReadVariableOp�
!Generator/covtr3/conv2d_transposeConv2DBackpropInputGenerator/covtr3/stack:output:08Generator/covtr3/conv2d_transpose/ReadVariableOp:value:02Generator/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������4,2*
paddingSAME*
strides
2#
!Generator/covtr3/conv2d_transpose�
'Generator/covtr3/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02)
'Generator/covtr3/BiasAdd/ReadVariableOp�
Generator/covtr3/BiasAddBiasAdd*Generator/covtr3/conv2d_transpose:output:0/Generator/covtr3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������4,22
Generator/covtr3/BiasAdd�
!Generator/leaky_re_lu_1/LeakyRelu	LeakyRelu!Generator/covtr3/BiasAdd:output:0*/
_output_shapes
:���������4,22#
!Generator/leaky_re_lu_1/LeakyRelu�
.Generator/batch_normalization_1/ReadVariableOpReadVariableOp7generator_batch_normalization_1_readvariableop_resource*
_output_shapes
:2*
dtype020
.Generator/batch_normalization_1/ReadVariableOp�
0Generator/batch_normalization_1/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:2*
dtype022
0Generator/batch_normalization_1/ReadVariableOp_1�
?Generator/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:2*
dtype02A
?Generator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
AGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:2*
dtype02C
AGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
0Generator/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_1/LeakyRelu:activations:06Generator/batch_normalization_1/ReadVariableOp:value:08Generator/batch_normalization_1/ReadVariableOp_1:value:0GGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������4,2:2:2:2:2:*
epsilon%o�:*
is_training( 22
0Generator/batch_normalization_1/FusedBatchNormV3�
Generator/covtr4/ShapeShape4Generator/batch_normalization_1/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr4/Shape�
$Generator/covtr4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr4/strided_slice/stack�
&Generator/covtr4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr4/strided_slice/stack_1�
&Generator/covtr4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr4/strided_slice/stack_2�
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
Generator/covtr4/stack/3�
Generator/covtr4/stackPack'Generator/covtr4/strided_slice:output:0!Generator/covtr4/stack/1:output:0!Generator/covtr4/stack/2:output:0!Generator/covtr4/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr4/stack�
&Generator/covtr4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr4/strided_slice_1/stack�
(Generator/covtr4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr4/strided_slice_1/stack_1�
(Generator/covtr4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr4/strided_slice_1/stack_2�
 Generator/covtr4/strided_slice_1StridedSliceGenerator/covtr4/stack:output:0/Generator/covtr4/strided_slice_1/stack:output:01Generator/covtr4/strided_slice_1/stack_1:output:01Generator/covtr4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr4/strided_slice_1�
0Generator/covtr4/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:P2*
dtype022
0Generator/covtr4/conv2d_transpose/ReadVariableOp�
!Generator/covtr4/conv2d_transposeConv2DBackpropInputGenerator/covtr4/stack:output:08Generator/covtr4/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������hXP*
paddingSAME*
strides
2#
!Generator/covtr4/conv2d_transpose�
'Generator/covtr4/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr4_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02)
'Generator/covtr4/BiasAdd/ReadVariableOp�
Generator/covtr4/BiasAddBiasAdd*Generator/covtr4/conv2d_transpose:output:0/Generator/covtr4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������hXP2
Generator/covtr4/BiasAdd�
!Generator/leaky_re_lu_2/LeakyRelu	LeakyRelu!Generator/covtr4/BiasAdd:output:0*/
_output_shapes
:���������hXP2#
!Generator/leaky_re_lu_2/LeakyRelu�
.Generator/batch_normalization_2/ReadVariableOpReadVariableOp7generator_batch_normalization_2_readvariableop_resource*
_output_shapes
:P*
dtype020
.Generator/batch_normalization_2/ReadVariableOp�
0Generator/batch_normalization_2/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:P*
dtype022
0Generator/batch_normalization_2/ReadVariableOp_1�
?Generator/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02A
?Generator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
AGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02C
AGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
0Generator/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_2/LeakyRelu:activations:06Generator/batch_normalization_2/ReadVariableOp:value:08Generator/batch_normalization_2/ReadVariableOp_1:value:0GGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������hXP:P:P:P:P:*
epsilon%o�:*
is_training( 22
0Generator/batch_normalization_2/FusedBatchNormV3�
Generator/cov3/ShapeShape4Generator/batch_normalization_2/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/cov3/Shape�
"Generator/cov3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"Generator/cov3/strided_slice/stack�
$Generator/cov3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$Generator/cov3/strided_slice/stack_1�
$Generator/cov3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$Generator/cov3/strided_slice/stack_2�
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
Generator/cov3/stack/3�
Generator/cov3/stackPack%Generator/cov3/strided_slice:output:0Generator/cov3/stack/1:output:0Generator/cov3/stack/2:output:0Generator/cov3/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/cov3/stack�
$Generator/cov3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/cov3/strided_slice_1/stack�
&Generator/cov3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/cov3/strided_slice_1/stack_1�
&Generator/cov3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/cov3/strided_slice_1/stack_2�
Generator/cov3/strided_slice_1StridedSliceGenerator/cov3/stack:output:0-Generator/cov3/strided_slice_1/stack:output:0/Generator/cov3/strided_slice_1/stack_1:output:0/Generator/cov3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Generator/cov3/strided_slice_1�
.Generator/cov3/conv2d_transpose/ReadVariableOpReadVariableOp7generator_cov3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:P*
dtype020
.Generator/cov3/conv2d_transpose/ReadVariableOp�
Generator/cov3/conv2d_transposeConv2DBackpropInputGenerator/cov3/stack:output:06Generator/cov3/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������mY*
paddingVALID*
strides
2!
Generator/cov3/conv2d_transpose�
%Generator/cov3/BiasAdd/ReadVariableOpReadVariableOp.generator_cov3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Generator/cov3/BiasAdd/ReadVariableOp�
Generator/cov3/BiasAddBiasAdd(Generator/cov3/conv2d_transpose:output:0-Generator/cov3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������mY2
Generator/cov3/BiasAdd�
Generator/cov3/SigmoidSigmoidGenerator/cov3/BiasAdd:output:0*
T0*/
_output_shapes
:���������mY2
Generator/cov3/Sigmoidv
IdentityIdentityGenerator/cov3/Sigmoid:y:0*
T0*/
_output_shapes
:���������mY2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d:::::::::::::::::::::::R N
'
_output_shapes
:���������d
#
_user_specified_name	gen_noise
�
�
*__inference_Generator_layer_call_fn_142015
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
identity��StatefulPartitionedCall�
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
-:+���������������������������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_Generator_layer_call_and_return_conditional_losses_1419682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:���������d
#
_user_specified_name	gen_noise
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_142634

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%��L>2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
{
&__inference_dense_layer_call_fn_142585

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1416422
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
c
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_141690

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+��������������������������� 2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+��������������������������� :i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�%
�
@__inference_cov3_layer_call_and_return_conditional_losses_141617

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity�D
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
strided_slice/stack_2�
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
strided_slice_1/stack_2�
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
strided_slice_2/stack_2�
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
stack/3�
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
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:P*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2	
Sigmoidy
IdentityIdentitySigmoid:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������P:::i e
A
_output_shapes/
-:+���������������������������P
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_141419

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:2*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:2*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:2*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:2*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������2:2:2:2:2:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������22

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������2:::::i e
A
_output_shapes/
-:+���������������������������2
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_141536

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������P:P:P:P:P:*
epsilon%o�:*
exponential_avg_factor%��L>2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+���������������������������P2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������P::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������P
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_142726

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:2*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:2*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:2*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:2*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������2:2:2:2:2:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������22

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������2:::::i e
A
_output_shapes/
-:+���������������������������2
 
_user_specified_nameinputs
�
z
%__inference_cov3_layer_call_fn_141627

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_cov3_layer_call_and_return_conditional_losses_1416172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������P::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������P
 
_user_specified_nameinputs
�
�
A__inference_dense_layer_call_and_return_conditional_losses_141642

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
*__inference_Generator_layer_call_fn_142124
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
identity��StatefulPartitionedCall�
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
-:+���������������������������*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_Generator_layer_call_and_return_conditional_losses_1420772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:���������d
#
_user_specified_name	gen_noise
�
|
'__inference_covtr3_layer_call_fn_141326

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr3_layer_call_and_return_conditional_losses_1413162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������22

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_142782

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������P:P:P:P:P:*
epsilon%o�:*
exponential_avg_factor%��L>2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+���������������������������P2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������P::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������P
 
_user_specified_nameinputs
�
e
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_142757

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������P2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������P2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������P:i e
A
_output_shapes/
-:+���������������������������P
 
_user_specified_nameinputs
�
|
'__inference_covtr2_layer_call_fn_141178

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr2_layer_call_and_return_conditional_losses_1411682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_141388

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:2*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:2*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:2*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:2*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������2:2:2:2:2:*
epsilon%o�:*
exponential_avg_factor%��L>2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+���������������������������22

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������2::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������2
 
_user_specified_nameinputs
�
H
,__inference_leaky_re_lu_layer_call_fn_142614

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_1416902
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+��������������������������� :i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
e
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_141743

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������22
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������2:i e
A
_output_shapes/
-:+���������������������������2
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_142800

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:P*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:P*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������P:P:P:P:P:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������P2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������P:::::i e
A
_output_shapes/
-:+���������������������������P
 
_user_specified_nameinputs
�
J
.__inference_leaky_re_lu_1_layer_call_fn_142688

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_1417432
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+���������������������������22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������2:i e
A
_output_shapes/
-:+���������������������������2
 
_user_specified_nameinputs
�
e
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_141796

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������P2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������P2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������P:i e
A
_output_shapes/
-:+���������������������������P
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_1_layer_call_fn_142752

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������2*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1414192
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������22

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������2::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������2
 
_user_specified_nameinputs
�
D
(__inference_reshape_layer_call_fn_142604

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1416722
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�!
�
B__inference_covtr3_layer_call_and_return_conditional_losses_141316

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity�D
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
strided_slice/stack_2�
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
strided_slice_1/stack_2�
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
strided_slice_2/stack_2�
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
stack/3�
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
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:2 *
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������2*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������22	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������22

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� :::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
��
�
E__inference_Generator_layer_call_and_return_conditional_losses_142324

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
identity��"batch_normalization/AssignNewValue�$batch_normalization/AssignNewValue_1�$batch_normalization_1/AssignNewValue�&batch_normalization_1/AssignNewValue_1�$batch_normalization_2/AssignNewValue�&batch_normalization_2/AssignNewValue_1�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2

dense/Reluf
reshape/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape�
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack�
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1�
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2�
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
reshape/Reshape/shape/3�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape�
reshape/ReshapeReshapedense/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2
reshape/Reshaped
covtr2/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
covtr2/Shape�
covtr2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr2/strided_slice/stack�
covtr2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr2/strided_slice/stack_1�
covtr2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr2/strided_slice/stack_2�
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
covtr2/stack/3�
covtr2/stackPackcovtr2/strided_slice:output:0covtr2/stack/1:output:0covtr2/stack/2:output:0covtr2/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr2/stack�
covtr2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr2/strided_slice_1/stack�
covtr2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr2/strided_slice_1/stack_1�
covtr2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr2/strided_slice_1/stack_2�
covtr2/strided_slice_1StridedSlicecovtr2/stack:output:0%covtr2/strided_slice_1/stack:output:0'covtr2/strided_slice_1/stack_1:output:0'covtr2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr2/strided_slice_1�
&covtr2/conv2d_transpose/ReadVariableOpReadVariableOp/covtr2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02(
&covtr2/conv2d_transpose/ReadVariableOp�
covtr2/conv2d_transposeConv2DBackpropInputcovtr2/stack:output:0.covtr2/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
covtr2/conv2d_transpose�
covtr2/BiasAdd/ReadVariableOpReadVariableOp&covtr2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
covtr2/BiasAdd/ReadVariableOp�
covtr2/BiasAddBiasAdd covtr2/conv2d_transpose:output:0%covtr2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
covtr2/BiasAdd�
leaky_re_lu/LeakyRelu	LeakyRelucovtr2/BiasAdd:output:0*/
_output_shapes
:��������� 2
leaky_re_lu/LeakyRelu�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3#leaky_re_lu/LeakyRelu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
exponential_avg_factor%��L>2&
$batch_normalization/FusedBatchNormV3�
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue�
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
covtr3/Shape�
covtr3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr3/strided_slice/stack�
covtr3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr3/strided_slice/stack_1�
covtr3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr3/strided_slice/stack_2�
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
covtr3/stack/3�
covtr3/stackPackcovtr3/strided_slice:output:0covtr3/stack/1:output:0covtr3/stack/2:output:0covtr3/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr3/stack�
covtr3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr3/strided_slice_1/stack�
covtr3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr3/strided_slice_1/stack_1�
covtr3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr3/strided_slice_1/stack_2�
covtr3/strided_slice_1StridedSlicecovtr3/stack:output:0%covtr3/strided_slice_1/stack:output:0'covtr3/strided_slice_1/stack_1:output:0'covtr3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr3/strided_slice_1�
&covtr3/conv2d_transpose/ReadVariableOpReadVariableOp/covtr3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:2 *
dtype02(
&covtr3/conv2d_transpose/ReadVariableOp�
covtr3/conv2d_transposeConv2DBackpropInputcovtr3/stack:output:0.covtr3/conv2d_transpose/ReadVariableOp:value:0(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������4,2*
paddingSAME*
strides
2
covtr3/conv2d_transpose�
covtr3/BiasAdd/ReadVariableOpReadVariableOp&covtr3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
covtr3/BiasAdd/ReadVariableOp�
covtr3/BiasAddBiasAdd covtr3/conv2d_transpose:output:0%covtr3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������4,22
covtr3/BiasAdd�
leaky_re_lu_1/LeakyRelu	LeakyRelucovtr3/BiasAdd:output:0*/
_output_shapes
:���������4,22
leaky_re_lu_1/LeakyRelu�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:2*
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:2*
dtype02(
&batch_normalization_1/ReadVariableOp_1�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:2*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:2*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_1/LeakyRelu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������4,2:2:2:2:2:*
epsilon%o�:*
exponential_avg_factor%��L>2(
&batch_normalization_1/FusedBatchNormV3�
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue�
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
covtr4/Shape�
covtr4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr4/strided_slice/stack�
covtr4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr4/strided_slice/stack_1�
covtr4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr4/strided_slice/stack_2�
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
covtr4/stack/3�
covtr4/stackPackcovtr4/strided_slice:output:0covtr4/stack/1:output:0covtr4/stack/2:output:0covtr4/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr4/stack�
covtr4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr4/strided_slice_1/stack�
covtr4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr4/strided_slice_1/stack_1�
covtr4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr4/strided_slice_1/stack_2�
covtr4/strided_slice_1StridedSlicecovtr4/stack:output:0%covtr4/strided_slice_1/stack:output:0'covtr4/strided_slice_1/stack_1:output:0'covtr4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr4/strided_slice_1�
&covtr4/conv2d_transpose/ReadVariableOpReadVariableOp/covtr4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:P2*
dtype02(
&covtr4/conv2d_transpose/ReadVariableOp�
covtr4/conv2d_transposeConv2DBackpropInputcovtr4/stack:output:0.covtr4/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������hXP*
paddingSAME*
strides
2
covtr4/conv2d_transpose�
covtr4/BiasAdd/ReadVariableOpReadVariableOp&covtr4_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
covtr4/BiasAdd/ReadVariableOp�
covtr4/BiasAddBiasAdd covtr4/conv2d_transpose:output:0%covtr4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������hXP2
covtr4/BiasAdd�
leaky_re_lu_2/LeakyRelu	LeakyRelucovtr4/BiasAdd:output:0*/
_output_shapes
:���������hXP2
leaky_re_lu_2/LeakyRelu�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:P*
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:P*
dtype02(
&batch_normalization_2/ReadVariableOp_1�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:P*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:P*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_2/LeakyRelu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������hXP:P:P:P:P:*
epsilon%o�:*
exponential_avg_factor%��L>2(
&batch_normalization_2/FusedBatchNormV3�
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue�
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
cov3/strided_slice/stack�
cov3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice/stack_1�
cov3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice/stack_2�
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
cov3/stack/3�

cov3/stackPackcov3/strided_slice:output:0cov3/stack/1:output:0cov3/stack/2:output:0cov3/stack/3:output:0*
N*
T0*
_output_shapes
:2

cov3/stack�
cov3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cov3/strided_slice_1/stack�
cov3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice_1/stack_1�
cov3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cov3/strided_slice_1/stack_2�
cov3/strided_slice_1StridedSlicecov3/stack:output:0#cov3/strided_slice_1/stack:output:0%cov3/strided_slice_1/stack_1:output:0%cov3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cov3/strided_slice_1�
$cov3/conv2d_transpose/ReadVariableOpReadVariableOp-cov3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:P*
dtype02&
$cov3/conv2d_transpose/ReadVariableOp�
cov3/conv2d_transposeConv2DBackpropInputcov3/stack:output:0,cov3/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������mY*
paddingVALID*
strides
2
cov3/conv2d_transpose�
cov3/BiasAdd/ReadVariableOpReadVariableOp$cov3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
cov3/BiasAdd/ReadVariableOp�
cov3/BiasAddBiasAddcov3/conv2d_transpose:output:0#cov3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������mY2
cov3/BiasAddx
cov3/SigmoidSigmoidcov3/BiasAdd:output:0*
T0*/
_output_shapes
:���������mY2
cov3/Sigmoid�
IdentityIdentitycov3/Sigmoid:y:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1*
T0*/
_output_shapes
:���������mY2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_1:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
e
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_142683

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������22
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������2:i e
A
_output_shapes/
-:+���������������������������2
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
	gen_noise2
serving_default_gen_noise:0���������d@
cov38
StatefulPartitionedCall:0���������mYtensorflow/serving/predict:۔
�q
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
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"�m
_tf_keras_network�m{"class_name": "Functional", "name": "Generator", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Generator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gen_noise"}, "name": "gen_noise", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 429, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["gen_noise", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [13, 11, 3]}}, "name": "reshape", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr2", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu", "inbound_nodes": [[["covtr2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr3", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr3", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_1", "inbound_nodes": [[["covtr3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr4", "trainable": true, "dtype": "float32", "filters": 80, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr4", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_2", "inbound_nodes": [[["covtr4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "cov3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "cov3", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}], "input_layers": [["gen_noise", 0, 0]], "output_layers": [["cov3", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Generator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gen_noise"}, "name": "gen_noise", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 429, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["gen_noise", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [13, 11, 3]}}, "name": "reshape", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr2", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu", "inbound_nodes": [[["covtr2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr3", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr3", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_1", "inbound_nodes": [[["covtr3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr4", "trainable": true, "dtype": "float32", "filters": 80, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr4", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_2", "inbound_nodes": [[["covtr4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "cov3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "cov3", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}], "input_layers": [["gen_noise", 0, 0]], "output_layers": [["cov3", 0, 0]]}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "gen_noise", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gen_noise"}}
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 429, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
�
	variables
regularization_losses
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [13, 11, 3]}}}
�


kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "covtr2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 11, 3]}}
�
#	variables
$regularization_losses
%trainable_variables
&	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
�	
'axis
	(gamma
)beta
*moving_mean
+moving_variance
,	variables
-regularization_losses
.trainable_variables
/	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 22, 32]}}
�


0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "covtr3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr3", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 22, 32]}}
�
6	variables
7regularization_losses
8trainable_variables
9	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
�	
:axis
	;gamma
<beta
=moving_mean
>moving_variance
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 52, 44, 50]}}
�


Ckernel
Dbias
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "covtr4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr4", "trainable": true, "dtype": "float32", "filters": 80, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 52, 44, 50]}}
�
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
�	
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 80}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 104, 88, 80]}}
�


Vkernel
Wbias
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "cov3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "cov3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 80}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 104, 88, 80]}}
 "
trackable_list_wrapper
�
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
�
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
�
\non_trainable_variables

]layers
^layer_regularization_losses
_metrics
regularization_losses
	variables
`layer_metrics
trainable_variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
:	d�2dense/kernel
:�2
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
�
anon_trainable_variables
	variables
blayer_regularization_losses
cmetrics
regularization_losses

dlayers
elayer_metrics
trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
fnon_trainable_variables
	variables
glayer_regularization_losses
hmetrics
regularization_losses

ilayers
jlayer_metrics
trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
knon_trainable_variables
	variables
llayer_regularization_losses
mmetrics
 regularization_losses

nlayers
olayer_metrics
!trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
pnon_trainable_variables
#	variables
qlayer_regularization_losses
rmetrics
$regularization_losses

slayers
tlayer_metrics
%trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
unon_trainable_variables
,	variables
vlayer_regularization_losses
wmetrics
-regularization_losses

xlayers
ylayer_metrics
.trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
znon_trainable_variables
2	variables
{layer_regularization_losses
|metrics
3regularization_losses

}layers
~layer_metrics
4trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables
6	variables
 �layer_regularization_losses
�metrics
7regularization_losses
�layers
�layer_metrics
8trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
�non_trainable_variables
?	variables
 �layer_regularization_losses
�metrics
@regularization_losses
�layers
�layer_metrics
Atrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
�non_trainable_variables
E	variables
 �layer_regularization_losses
�metrics
Fregularization_losses
�layers
�layer_metrics
Gtrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
I	variables
 �layer_regularization_losses
�metrics
Jregularization_losses
�layers
�layer_metrics
Ktrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
�non_trainable_variables
R	variables
 �layer_regularization_losses
�metrics
Sregularization_losses
�layers
�layer_metrics
Ttrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
�non_trainable_variables
X	variables
 �layer_regularization_losses
�metrics
Yregularization_losses
�layers
�layer_metrics
Ztrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�2�
*__inference_Generator_layer_call_fn_142516
*__inference_Generator_layer_call_fn_142124
*__inference_Generator_layer_call_fn_142015
*__inference_Generator_layer_call_fn_142565�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_Generator_layer_call_and_return_conditional_losses_142467
E__inference_Generator_layer_call_and_return_conditional_losses_141845
E__inference_Generator_layer_call_and_return_conditional_losses_142324
E__inference_Generator_layer_call_and_return_conditional_losses_141905�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_141134�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *(�%
#� 
	gen_noise���������d
�2�
A__inference_dense_layer_call_and_return_conditional_losses_142576�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_dense_layer_call_fn_142585�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_reshape_layer_call_and_return_conditional_losses_142599�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_reshape_layer_call_fn_142604�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_covtr2_layer_call_and_return_conditional_losses_141168�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
'__inference_covtr2_layer_call_fn_141178�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_142609�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_leaky_re_lu_layer_call_fn_142614�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_142634
O__inference_batch_normalization_layer_call_and_return_conditional_losses_142652�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
4__inference_batch_normalization_layer_call_fn_142665
4__inference_batch_normalization_layer_call_fn_142678�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_covtr3_layer_call_and_return_conditional_losses_141316�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
'__inference_covtr3_layer_call_fn_141326�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_142683�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_leaky_re_lu_1_layer_call_fn_142688�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_142726
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_142708�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
6__inference_batch_normalization_1_layer_call_fn_142739
6__inference_batch_normalization_1_layer_call_fn_142752�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_covtr4_layer_call_and_return_conditional_losses_141464�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������2
�2�
'__inference_covtr4_layer_call_fn_141474�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������2
�2�
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_142757�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_leaky_re_lu_2_layer_call_fn_142762�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_142800
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_142782�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
6__inference_batch_normalization_2_layer_call_fn_142813
6__inference_batch_normalization_2_layer_call_fn_142826�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
@__inference_cov3_layer_call_and_return_conditional_losses_141617�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������P
�2�
%__inference_cov3_layer_call_fn_141627�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������P
5B3
$__inference_signature_wrapper_142175	gen_noise�
E__inference_Generator_layer_call_and_return_conditional_losses_141845�()*+01;<=>CDNOPQVW:�7
0�-
#� 
	gen_noise���������d
p

 
� "?�<
5�2
0+���������������������������
� �
E__inference_Generator_layer_call_and_return_conditional_losses_141905�()*+01;<=>CDNOPQVW:�7
0�-
#� 
	gen_noise���������d
p 

 
� "?�<
5�2
0+���������������������������
� �
E__inference_Generator_layer_call_and_return_conditional_losses_142324�()*+01;<=>CDNOPQVW7�4
-�*
 �
inputs���������d
p

 
� "-�*
#� 
0���������mY
� �
E__inference_Generator_layer_call_and_return_conditional_losses_142467�()*+01;<=>CDNOPQVW7�4
-�*
 �
inputs���������d
p 

 
� "-�*
#� 
0���������mY
� �
*__inference_Generator_layer_call_fn_142015�()*+01;<=>CDNOPQVW:�7
0�-
#� 
	gen_noise���������d
p

 
� "2�/+����������������������������
*__inference_Generator_layer_call_fn_142124�()*+01;<=>CDNOPQVW:�7
0�-
#� 
	gen_noise���������d
p 

 
� "2�/+����������������������������
*__inference_Generator_layer_call_fn_142516�()*+01;<=>CDNOPQVW7�4
-�*
 �
inputs���������d
p

 
� "2�/+����������������������������
*__inference_Generator_layer_call_fn_142565�()*+01;<=>CDNOPQVW7�4
-�*
 �
inputs���������d
p 

 
� "2�/+����������������������������
!__inference__wrapped_model_141134�()*+01;<=>CDNOPQVW2�/
(�%
#� 
	gen_noise���������d
� "3�0
.
cov3&�#
cov3���������mY�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_142708�;<=>M�J
C�@
:�7
inputs+���������������������������2
p
� "?�<
5�2
0+���������������������������2
� �
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_142726�;<=>M�J
C�@
:�7
inputs+���������������������������2
p 
� "?�<
5�2
0+���������������������������2
� �
6__inference_batch_normalization_1_layer_call_fn_142739�;<=>M�J
C�@
:�7
inputs+���������������������������2
p
� "2�/+���������������������������2�
6__inference_batch_normalization_1_layer_call_fn_142752�;<=>M�J
C�@
:�7
inputs+���������������������������2
p 
� "2�/+���������������������������2�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_142782�NOPQM�J
C�@
:�7
inputs+���������������������������P
p
� "?�<
5�2
0+���������������������������P
� �
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_142800�NOPQM�J
C�@
:�7
inputs+���������������������������P
p 
� "?�<
5�2
0+���������������������������P
� �
6__inference_batch_normalization_2_layer_call_fn_142813�NOPQM�J
C�@
:�7
inputs+���������������������������P
p
� "2�/+���������������������������P�
6__inference_batch_normalization_2_layer_call_fn_142826�NOPQM�J
C�@
:�7
inputs+���������������������������P
p 
� "2�/+���������������������������P�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_142634�()*+M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
O__inference_batch_normalization_layer_call_and_return_conditional_losses_142652�()*+M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
4__inference_batch_normalization_layer_call_fn_142665�()*+M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
4__inference_batch_normalization_layer_call_fn_142678�()*+M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
@__inference_cov3_layer_call_and_return_conditional_losses_141617�VWI�F
?�<
:�7
inputs+���������������������������P
� "?�<
5�2
0+���������������������������
� �
%__inference_cov3_layer_call_fn_141627�VWI�F
?�<
:�7
inputs+���������������������������P
� "2�/+����������������������������
B__inference_covtr2_layer_call_and_return_conditional_losses_141168�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+��������������������������� 
� �
'__inference_covtr2_layer_call_fn_141178�I�F
?�<
:�7
inputs+���������������������������
� "2�/+��������������������������� �
B__inference_covtr3_layer_call_and_return_conditional_losses_141316�01I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������2
� �
'__inference_covtr3_layer_call_fn_141326�01I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+���������������������������2�
B__inference_covtr4_layer_call_and_return_conditional_losses_141464�CDI�F
?�<
:�7
inputs+���������������������������2
� "?�<
5�2
0+���������������������������P
� �
'__inference_covtr4_layer_call_fn_141474�CDI�F
?�<
:�7
inputs+���������������������������2
� "2�/+���������������������������P�
A__inference_dense_layer_call_and_return_conditional_losses_142576]/�,
%�"
 �
inputs���������d
� "&�#
�
0����������
� z
&__inference_dense_layer_call_fn_142585P/�,
%�"
 �
inputs���������d
� "������������
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_142683�I�F
?�<
:�7
inputs+���������������������������2
� "?�<
5�2
0+���������������������������2
� �
.__inference_leaky_re_lu_1_layer_call_fn_142688I�F
?�<
:�7
inputs+���������������������������2
� "2�/+���������������������������2�
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_142757�I�F
?�<
:�7
inputs+���������������������������P
� "?�<
5�2
0+���������������������������P
� �
.__inference_leaky_re_lu_2_layer_call_fn_142762I�F
?�<
:�7
inputs+���������������������������P
� "2�/+���������������������������P�
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_142609�I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+��������������������������� 
� �
,__inference_leaky_re_lu_layer_call_fn_142614I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+��������������������������� �
C__inference_reshape_layer_call_and_return_conditional_losses_142599a0�-
&�#
!�
inputs����������
� "-�*
#� 
0���������
� �
(__inference_reshape_layer_call_fn_142604T0�-
&�#
!�
inputs����������
� " �����������
$__inference_signature_wrapper_142175�()*+01;<=>CDNOPQVW?�<
� 
5�2
0
	gen_noise#� 
	gen_noise���������d"3�0
.
cov3&�#
cov3���������mY