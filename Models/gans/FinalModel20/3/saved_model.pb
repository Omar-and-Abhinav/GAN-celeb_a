�"
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
 �"serve*2.3.12v2.3.0-54-gfcc4b966f18��
~
covtr4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr4/kernel
w
!covtr4/kernel/Read/ReadVariableOpReadVariableOpcovtr4/kernel*&
_output_shapes
:*
dtype0
n
covtr4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr4/bias
g
covtr4/bias/Read/ReadVariableOpReadVariableOpcovtr4/bias*
_output_shapes
:*
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
~
covtr5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr5/kernel
w
!covtr5/kernel/Read/ReadVariableOpReadVariableOpcovtr5/kernel*&
_output_shapes
:*
dtype0
n
covtr5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr5/bias
g
covtr5/bias/Read/ReadVariableOpReadVariableOpcovtr5/bias*
_output_shapes
:*
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
~
covtr6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr6/kernel
w
!covtr6/kernel/Read/ReadVariableOpReadVariableOpcovtr6/kernel*&
_output_shapes
:*
dtype0
n
covtr6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr6/bias
g
covtr6/bias/Read/ReadVariableOpReadVariableOpcovtr6/bias*
_output_shapes
:*
dtype0
�
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_2/gamma
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:*
dtype0
�
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_2/beta
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:*
dtype0
�
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_2/moving_mean
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:*
dtype0
�
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_2/moving_variance
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:*
dtype0
~
covtr7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr7/kernel
w
!covtr7/kernel/Read/ReadVariableOpReadVariableOpcovtr7/kernel*&
_output_shapes
:*
dtype0
n
covtr7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr7/bias
g
covtr7/bias/Read/ReadVariableOpReadVariableOpcovtr7/bias*
_output_shapes
:*
dtype0
�
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_3/gamma
�
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:*
dtype0
�
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_3/beta
�
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:*
dtype0
�
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_3/moving_mean
�
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:*
dtype0
�
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_3/moving_variance
�
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:*
dtype0
~
covtr8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr8/kernel
w
!covtr8/kernel/Read/ReadVariableOpReadVariableOpcovtr8/kernel*&
_output_shapes
:*
dtype0
n
covtr8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr8/bias
g
covtr8/bias/Read/ReadVariableOpReadVariableOpcovtr8/bias*
_output_shapes
:*
dtype0
�
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_4/gamma
�
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:*
dtype0
�
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_4/beta
�
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:*
dtype0
�
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_4/moving_mean
�
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:*
dtype0
�
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_4/moving_variance
�
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:*
dtype0
~
covtr9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr9/kernel
w
!covtr9/kernel/Read/ReadVariableOpReadVariableOpcovtr9/kernel*&
_output_shapes
:*
dtype0
n
covtr9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr9/bias
g
covtr9/bias/Read/ReadVariableOpReadVariableOpcovtr9/bias*
_output_shapes
:*
dtype0
�
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_5/gamma
�
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:*
dtype0
�
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_5/beta
�
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:*
dtype0
�
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_5/moving_mean
�
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:*
dtype0
�
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_5/moving_variance
�
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:*
dtype0
�
covtr10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr10/kernel
y
"covtr10/kernel/Read/ReadVariableOpReadVariableOpcovtr10/kernel*&
_output_shapes
:*
dtype0
p
covtr10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr10/bias
i
 covtr10/bias/Read/ReadVariableOpReadVariableOpcovtr10/bias*
_output_shapes
:*
dtype0
�
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_6/gamma
�
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
:*
dtype0
�
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_6/beta
�
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:*
dtype0
�
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_6/moving_mean
�
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:*
dtype0
�
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_6/moving_variance
�
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:*
dtype0
�
covtr11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		"*
shared_namecovtr11/kernel
y
"covtr11/kernel/Read/ReadVariableOpReadVariableOpcovtr11/kernel*&
_output_shapes
:		"*
dtype0
p
covtr11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:"*
shared_namecovtr11/bias
i
 covtr11/bias/Read/ReadVariableOpReadVariableOpcovtr11/bias*
_output_shapes
:"*
dtype0
�
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:"*,
shared_namebatch_normalization_7/gamma
�
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:"*
dtype0
�
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:"*+
shared_namebatch_normalization_7/beta
�
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:"*
dtype0
�
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:"*2
shared_name#!batch_normalization_7/moving_mean
�
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:"*
dtype0
�
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:"*6
shared_name'%batch_normalization_7/moving_variance
�
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:"*
dtype0
�
covtr14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:"*
shared_namecovtr14/kernel
y
"covtr14/kernel/Read/ReadVariableOpReadVariableOpcovtr14/kernel*&
_output_shapes
:"*
dtype0
p
covtr14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecovtr14/bias
i
 covtr14/bias/Read/ReadVariableOpReadVariableOpcovtr14/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B�~ B�~
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer-15
layer_with_weights-9
layer-16
layer_with_weights-10
layer-17
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer-21
layer_with_weights-13
layer-22
layer_with_weights-14
layer-23
layer-24
layer_with_weights-15
layer-25
layer_with_weights-16
layer-26
	variables
regularization_losses
trainable_variables
	keras_api
 
signatures
 
R
!	variables
"regularization_losses
#trainable_variables
$	keras_api
h

%kernel
&bias
'	variables
(regularization_losses
)trainable_variables
*	keras_api
R
+	variables
,regularization_losses
-trainable_variables
.	keras_api
�
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4	variables
5regularization_losses
6trainable_variables
7	keras_api
h

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
R
>	variables
?regularization_losses
@trainable_variables
A	keras_api
�
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
h

Kkernel
Lbias
M	variables
Nregularization_losses
Otrainable_variables
P	keras_api
R
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api
�
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
h

^kernel
_bias
`	variables
aregularization_losses
btrainable_variables
c	keras_api
R
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
�
haxis
	igamma
jbeta
kmoving_mean
lmoving_variance
m	variables
nregularization_losses
otrainable_variables
p	keras_api
h

qkernel
rbias
s	variables
tregularization_losses
utrainable_variables
v	keras_api
R
w	variables
xregularization_losses
ytrainable_variables
z	keras_api
�
{axis
	|gamma
}beta
~moving_mean
moving_variance
�	variables
�regularization_losses
�trainable_variables
�	keras_api
n
�kernel
	�bias
�	variables
�regularization_losses
�trainable_variables
�	keras_api
V
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�	variables
�regularization_losses
�trainable_variables
�	keras_api
n
�kernel
	�bias
�	variables
�regularization_losses
�trainable_variables
�	keras_api
V
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�	variables
�regularization_losses
�trainable_variables
�	keras_api
n
�kernel
	�bias
�	variables
�regularization_losses
�trainable_variables
�	keras_api
V
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�	variables
�regularization_losses
�trainable_variables
�	keras_api
n
�kernel
	�bias
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�
%0
&1
02
13
24
35
86
97
C8
D9
E10
F11
K12
L13
V14
W15
X16
Y17
^18
_19
i20
j21
k22
l23
q24
r25
|26
}27
~28
29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
 
�
%0
&1
02
13
84
95
C6
D7
K8
L9
V10
W11
^12
_13
i14
j15
q16
r17
|18
}19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�
	variables
�non_trainable_variables
regularization_losses
 �layer_regularization_losses
�layers
trainable_variables
�layer_metrics
�metrics
 
 
 
 
�
!	variables
�non_trainable_variables
"regularization_losses
 �layer_regularization_losses
�layers
#trainable_variables
�layer_metrics
�metrics
YW
VARIABLE_VALUEcovtr4/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr4/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1
 

%0
&1
�
'	variables
�non_trainable_variables
(regularization_losses
 �layer_regularization_losses
�layers
)trainable_variables
�layer_metrics
�metrics
 
 
 
�
+	variables
�non_trainable_variables
,regularization_losses
 �layer_regularization_losses
�layers
-trainable_variables
�layer_metrics
�metrics
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

00
11
22
33
 

00
11
�
4	variables
�non_trainable_variables
5regularization_losses
 �layer_regularization_losses
�layers
6trainable_variables
�layer_metrics
�metrics
YW
VARIABLE_VALUEcovtr5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91
 

80
91
�
:	variables
�non_trainable_variables
;regularization_losses
 �layer_regularization_losses
�layers
<trainable_variables
�layer_metrics
�metrics
 
 
 
�
>	variables
�non_trainable_variables
?regularization_losses
 �layer_regularization_losses
�layers
@trainable_variables
�layer_metrics
�metrics
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

C0
D1
E2
F3
 

C0
D1
�
G	variables
�non_trainable_variables
Hregularization_losses
 �layer_regularization_losses
�layers
Itrainable_variables
�layer_metrics
�metrics
YW
VARIABLE_VALUEcovtr6/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr6/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

K0
L1
 

K0
L1
�
M	variables
�non_trainable_variables
Nregularization_losses
 �layer_regularization_losses
�layers
Otrainable_variables
�layer_metrics
�metrics
 
 
 
�
Q	variables
�non_trainable_variables
Rregularization_losses
 �layer_regularization_losses
�layers
Strainable_variables
�layer_metrics
�metrics
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

V0
W1
X2
Y3
 

V0
W1
�
Z	variables
�non_trainable_variables
[regularization_losses
 �layer_regularization_losses
�layers
\trainable_variables
�layer_metrics
�metrics
YW
VARIABLE_VALUEcovtr7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr7/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

^0
_1
 

^0
_1
�
`	variables
�non_trainable_variables
aregularization_losses
 �layer_regularization_losses
�layers
btrainable_variables
�layer_metrics
�metrics
 
 
 
�
d	variables
�non_trainable_variables
eregularization_losses
 �layer_regularization_losses
�layers
ftrainable_variables
�layer_metrics
�metrics
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

i0
j1
k2
l3
 

i0
j1
�
m	variables
�non_trainable_variables
nregularization_losses
 �layer_regularization_losses
�layers
otrainable_variables
�layer_metrics
�metrics
YW
VARIABLE_VALUEcovtr8/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr8/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

q0
r1
 

q0
r1
�
s	variables
�non_trainable_variables
tregularization_losses
 �layer_regularization_losses
�layers
utrainable_variables
�layer_metrics
�metrics
 
 
 
�
w	variables
�non_trainable_variables
xregularization_losses
 �layer_regularization_losses
�layers
ytrainable_variables
�layer_metrics
�metrics
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

|0
}1
~2
3
 

|0
}1
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
ZX
VARIABLE_VALUEcovtr9/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEcovtr9/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
 
 
 
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
 
ge
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
�0
�1
�2
�3
 

�0
�1
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
[Y
VARIABLE_VALUEcovtr10/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEcovtr10/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
 
 
 
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
 
ge
VARIABLE_VALUEbatch_normalization_6/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_6/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_6/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_6/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
�0
�1
�2
�3
 

�0
�1
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
[Y
VARIABLE_VALUEcovtr11/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEcovtr11/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
 
 
 
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
 
ge
VARIABLE_VALUEbatch_normalization_7/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_7/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_7/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_7/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
�0
�1
�2
�3
 

�0
�1
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
[Y
VARIABLE_VALUEcovtr14/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEcovtr14/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
|
20
31
E2
F3
X4
Y5
k6
l7
~8
9
�10
�11
�12
�13
�14
�15
 
�
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
16
17
18
19
20
21
22
23
24
25
26
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
20
31
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
E0
F1
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
X0
Y1
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
k0
l1
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
~0
1
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

�0
�1
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

�0
�1
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

�0
�1
 
 
 
 
 
 
 
 
 
~
serving_default_gen_noisePlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_gen_noisecovtr4/kernelcovtr4/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancecovtr5/kernelcovtr5/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancecovtr6/kernelcovtr6/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancecovtr7/kernelcovtr7/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancecovtr8/kernelcovtr8/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancecovtr9/kernelcovtr9/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variancecovtr10/kernelcovtr10/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancecovtr11/kernelcovtr11/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancecovtr14/kernelcovtr14/bias*>
Tin7
523*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������7-*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*2
config_proto" 

CPU

GPU2 *0J 8� *,
f'R%
#__inference_signature_wrapper_94954
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!covtr4/kernel/Read/ReadVariableOpcovtr4/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp!covtr5/kernel/Read/ReadVariableOpcovtr5/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp!covtr6/kernel/Read/ReadVariableOpcovtr6/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp!covtr7/kernel/Read/ReadVariableOpcovtr7/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp!covtr8/kernel/Read/ReadVariableOpcovtr8/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp!covtr9/kernel/Read/ReadVariableOpcovtr9/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp"covtr10/kernel/Read/ReadVariableOp covtr10/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp"covtr11/kernel/Read/ReadVariableOp covtr11/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp"covtr14/kernel/Read/ReadVariableOp covtr14/bias/Read/ReadVariableOpConst*?
Tin8
624*
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
GPU2 *0J 8� *'
f"R 
__inference__traced_save_96576
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecovtr4/kernelcovtr4/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancecovtr5/kernelcovtr5/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancecovtr6/kernelcovtr6/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancecovtr7/kernelcovtr7/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancecovtr8/kernelcovtr8/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancecovtr9/kernelcovtr9/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variancecovtr10/kernelcovtr10/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancecovtr11/kernelcovtr11/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancecovtr14/kernelcovtr14/bias*>
Tin7
523*
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
GPU2 *0J 8� **
f%R#
!__inference__traced_restore_96736أ
�
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_96063

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
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
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
|
'__inference_covtr10_layer_call_fn_93485

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
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr10_layer_call_and_return_conditional_losses_934752
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_92635

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
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
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
{
&__inference_covtr4_layer_call_fn_92573

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
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr4_layer_call_and_return_conditional_losses_925632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
^
B__inference_reshape_layer_call_and_return_conditional_losses_93808

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
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
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
:���������2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_Generator_layer_call_fn_95687

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

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48
identity��StatefulPartitionedCall�
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
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*D
_read_only_resource_inputs&
$"	
 !"%&'(+,-.12*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_Generator_layer_call_and_return_conditional_losses_945092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_93274

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_96285

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
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
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_1_layer_call_fn_95946

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
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_927872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
3__inference_batch_normalization_layer_call_fn_95872

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
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_926352
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
{
&__inference_covtr5_layer_call_fn_92725

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
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr5_layer_call_and_return_conditional_losses_927152
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
��
�
D__inference_Generator_layer_call_and_return_conditional_losses_94376
	gen_noise
covtr4_94250
covtr4_94252
batch_normalization_94256
batch_normalization_94258
batch_normalization_94260
batch_normalization_94262
covtr5_94265
covtr5_94267
batch_normalization_1_94271
batch_normalization_1_94273
batch_normalization_1_94275
batch_normalization_1_94277
covtr6_94280
covtr6_94282
batch_normalization_2_94286
batch_normalization_2_94288
batch_normalization_2_94290
batch_normalization_2_94292
covtr7_94295
covtr7_94297
batch_normalization_3_94301
batch_normalization_3_94303
batch_normalization_3_94305
batch_normalization_3_94307
covtr8_94310
covtr8_94312
batch_normalization_4_94316
batch_normalization_4_94318
batch_normalization_4_94320
batch_normalization_4_94322
covtr9_94325
covtr9_94327
batch_normalization_5_94331
batch_normalization_5_94333
batch_normalization_5_94335
batch_normalization_5_94337
covtr10_94340
covtr10_94342
batch_normalization_6_94346
batch_normalization_6_94348
batch_normalization_6_94350
batch_normalization_6_94352
covtr11_94355
covtr11_94357
batch_normalization_7_94361
batch_normalization_7_94363
batch_normalization_7_94365
batch_normalization_7_94367
covtr14_94370
covtr14_94372
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�covtr10/StatefulPartitionedCall�covtr11/StatefulPartitionedCall�covtr14/StatefulPartitionedCall�covtr4/StatefulPartitionedCall�covtr5/StatefulPartitionedCall�covtr6/StatefulPartitionedCall�covtr7/StatefulPartitionedCall�covtr8/StatefulPartitionedCall�covtr9/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCall	gen_noise*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_938082
reshape/PartitionedCall�
covtr4/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr4_94250covtr4_94252*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr4_layer_call_and_return_conditional_losses_925632 
covtr4/StatefulPartitionedCall�
leaky_re_lu/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_938262
leaky_re_lu/PartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_94256batch_normalization_94258batch_normalization_94260batch_normalization_94262*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_926662-
+batch_normalization/StatefulPartitionedCall�
covtr5/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr5_94265covtr5_94267*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr5_layer_call_and_return_conditional_losses_927152 
covtr5/StatefulPartitionedCall�
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_938792
leaky_re_lu_1/PartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_94271batch_normalization_1_94273batch_normalization_1_94275batch_normalization_1_94277*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_928182/
-batch_normalization_1/StatefulPartitionedCall�
covtr6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr6_94280covtr6_94282*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr6_layer_call_and_return_conditional_losses_928672 
covtr6/StatefulPartitionedCall�
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_939322
leaky_re_lu_2/PartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_94286batch_normalization_2_94288batch_normalization_2_94290batch_normalization_2_94292*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_929702/
-batch_normalization_2/StatefulPartitionedCall�
covtr7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0covtr7_94295covtr7_94297*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr7_layer_call_and_return_conditional_losses_930192 
covtr7/StatefulPartitionedCall�
leaky_re_lu_3/PartitionedCallPartitionedCall'covtr7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_939852
leaky_re_lu_3/PartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0batch_normalization_3_94301batch_normalization_3_94303batch_normalization_3_94305batch_normalization_3_94307*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_931222/
-batch_normalization_3/StatefulPartitionedCall�
covtr8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0covtr8_94310covtr8_94312*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr8_layer_call_and_return_conditional_losses_931712 
covtr8/StatefulPartitionedCall�
leaky_re_lu_4/PartitionedCallPartitionedCall'covtr8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_940382
leaky_re_lu_4/PartitionedCall�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0batch_normalization_4_94316batch_normalization_4_94318batch_normalization_4_94320batch_normalization_4_94322*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_932742/
-batch_normalization_4/StatefulPartitionedCall�
covtr9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0covtr9_94325covtr9_94327*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr9_layer_call_and_return_conditional_losses_933232 
covtr9/StatefulPartitionedCall�
leaky_re_lu_5/PartitionedCallPartitionedCall'covtr9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_940912
leaky_re_lu_5/PartitionedCall�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0batch_normalization_5_94331batch_normalization_5_94333batch_normalization_5_94335batch_normalization_5_94337*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_934262/
-batch_normalization_5/StatefulPartitionedCall�
covtr10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0covtr10_94340covtr10_94342*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr10_layer_call_and_return_conditional_losses_934752!
covtr10/StatefulPartitionedCall�
leaky_re_lu_6/PartitionedCallPartitionedCall(covtr10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_941442
leaky_re_lu_6/PartitionedCall�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0batch_normalization_6_94346batch_normalization_6_94348batch_normalization_6_94350batch_normalization_6_94352*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_935782/
-batch_normalization_6/StatefulPartitionedCall�
covtr11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0covtr11_94355covtr11_94357*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������"*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr11_layer_call_and_return_conditional_losses_936272!
covtr11/StatefulPartitionedCall�
leaky_re_lu_7/PartitionedCallPartitionedCall(covtr11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������"* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_941972
leaky_re_lu_7/PartitionedCall�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0batch_normalization_7_94361batch_normalization_7_94363batch_normalization_7_94365batch_normalization_7_94367*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������"*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_937302/
-batch_normalization_7/StatefulPartitionedCall�
covtr14/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0covtr14_94370covtr14_94372*
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
GPU2 *0J 8� *K
fFRD
B__inference_covtr14_layer_call_and_return_conditional_losses_937802!
covtr14/StatefulPartitionedCall�
IdentityIdentity(covtr14/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall ^covtr10/StatefulPartitionedCall ^covtr11/StatefulPartitionedCall ^covtr14/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^covtr5/StatefulPartitionedCall^covtr6/StatefulPartitionedCall^covtr7/StatefulPartitionedCall^covtr8/StatefulPartitionedCall^covtr9/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������::::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2B
covtr10/StatefulPartitionedCallcovtr10/StatefulPartitionedCall2B
covtr11/StatefulPartitionedCallcovtr11/StatefulPartitionedCall2B
covtr14/StatefulPartitionedCallcovtr14/StatefulPartitionedCall2@
covtr4/StatefulPartitionedCallcovtr4/StatefulPartitionedCall2@
covtr5/StatefulPartitionedCallcovtr5/StatefulPartitionedCall2@
covtr6/StatefulPartitionedCallcovtr6/StatefulPartitionedCall2@
covtr7/StatefulPartitionedCallcovtr7/StatefulPartitionedCall2@
covtr8/StatefulPartitionedCallcovtr8/StatefulPartitionedCall2@
covtr9/StatefulPartitionedCallcovtr9/StatefulPartitionedCall:S O
(
_output_shapes
:����������
#
_user_specified_name	gen_noise
�
I
-__inference_leaky_re_lu_2_layer_call_fn_95969

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
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_939322
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
��
�
!__inference__traced_restore_96736
file_prefix"
assignvariableop_covtr4_kernel"
assignvariableop_1_covtr4_bias0
,assignvariableop_2_batch_normalization_gamma/
+assignvariableop_3_batch_normalization_beta6
2assignvariableop_4_batch_normalization_moving_mean:
6assignvariableop_5_batch_normalization_moving_variance$
 assignvariableop_6_covtr5_kernel"
assignvariableop_7_covtr5_bias2
.assignvariableop_8_batch_normalization_1_gamma1
-assignvariableop_9_batch_normalization_1_beta9
5assignvariableop_10_batch_normalization_1_moving_mean=
9assignvariableop_11_batch_normalization_1_moving_variance%
!assignvariableop_12_covtr6_kernel#
assignvariableop_13_covtr6_bias3
/assignvariableop_14_batch_normalization_2_gamma2
.assignvariableop_15_batch_normalization_2_beta9
5assignvariableop_16_batch_normalization_2_moving_mean=
9assignvariableop_17_batch_normalization_2_moving_variance%
!assignvariableop_18_covtr7_kernel#
assignvariableop_19_covtr7_bias3
/assignvariableop_20_batch_normalization_3_gamma2
.assignvariableop_21_batch_normalization_3_beta9
5assignvariableop_22_batch_normalization_3_moving_mean=
9assignvariableop_23_batch_normalization_3_moving_variance%
!assignvariableop_24_covtr8_kernel#
assignvariableop_25_covtr8_bias3
/assignvariableop_26_batch_normalization_4_gamma2
.assignvariableop_27_batch_normalization_4_beta9
5assignvariableop_28_batch_normalization_4_moving_mean=
9assignvariableop_29_batch_normalization_4_moving_variance%
!assignvariableop_30_covtr9_kernel#
assignvariableop_31_covtr9_bias3
/assignvariableop_32_batch_normalization_5_gamma2
.assignvariableop_33_batch_normalization_5_beta9
5assignvariableop_34_batch_normalization_5_moving_mean=
9assignvariableop_35_batch_normalization_5_moving_variance&
"assignvariableop_36_covtr10_kernel$
 assignvariableop_37_covtr10_bias3
/assignvariableop_38_batch_normalization_6_gamma2
.assignvariableop_39_batch_normalization_6_beta9
5assignvariableop_40_batch_normalization_6_moving_mean=
9assignvariableop_41_batch_normalization_6_moving_variance&
"assignvariableop_42_covtr11_kernel$
 assignvariableop_43_covtr11_bias3
/assignvariableop_44_batch_normalization_7_gamma2
.assignvariableop_45_batch_normalization_7_beta9
5assignvariableop_46_batch_normalization_7_moving_mean=
9assignvariableop_47_batch_normalization_7_moving_variance&
"assignvariableop_48_covtr14_kernel$
 assignvariableop_49_covtr14_bias
identity_51��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*�
value�B�3B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::*A
dtypes7
5232
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_covtr4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_covtr4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp assignvariableop_6_covtr5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_covtr5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp!assignvariableop_12_covtr6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_covtr6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp!assignvariableop_18_covtr7_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_covtr7_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_3_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_3_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp!assignvariableop_24_covtr8_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOpassignvariableop_25_covtr8_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_4_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_4_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_4_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_4_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp!assignvariableop_30_covtr9_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOpassignvariableop_31_covtr9_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp/assignvariableop_32_batch_normalization_5_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp.assignvariableop_33_batch_normalization_5_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp5assignvariableop_34_batch_normalization_5_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp9assignvariableop_35_batch_normalization_5_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp"assignvariableop_36_covtr10_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp assignvariableop_37_covtr10_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp/assignvariableop_38_batch_normalization_6_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp.assignvariableop_39_batch_normalization_6_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp5assignvariableop_40_batch_normalization_6_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp9assignvariableop_41_batch_normalization_6_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp"assignvariableop_42_covtr11_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp assignvariableop_43_covtr11_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp/assignvariableop_44_batch_normalization_7_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp.assignvariableop_45_batch_normalization_7_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp5assignvariableop_46_batch_normalization_7_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp9assignvariableop_47_batch_normalization_7_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp"assignvariableop_48_covtr14_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp assignvariableop_49_covtr14_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_499
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�	
Identity_50Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_50�	
Identity_51IdentityIdentity_50:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_51"#
identity_51Identity_51:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
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
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_92666

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_92787

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
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
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_93547

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
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
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_93426

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
b
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_93826

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
��
�
D__inference_Generator_layer_call_and_return_conditional_losses_94744

inputs
covtr4_94618
covtr4_94620
batch_normalization_94624
batch_normalization_94626
batch_normalization_94628
batch_normalization_94630
covtr5_94633
covtr5_94635
batch_normalization_1_94639
batch_normalization_1_94641
batch_normalization_1_94643
batch_normalization_1_94645
covtr6_94648
covtr6_94650
batch_normalization_2_94654
batch_normalization_2_94656
batch_normalization_2_94658
batch_normalization_2_94660
covtr7_94663
covtr7_94665
batch_normalization_3_94669
batch_normalization_3_94671
batch_normalization_3_94673
batch_normalization_3_94675
covtr8_94678
covtr8_94680
batch_normalization_4_94684
batch_normalization_4_94686
batch_normalization_4_94688
batch_normalization_4_94690
covtr9_94693
covtr9_94695
batch_normalization_5_94699
batch_normalization_5_94701
batch_normalization_5_94703
batch_normalization_5_94705
covtr10_94708
covtr10_94710
batch_normalization_6_94714
batch_normalization_6_94716
batch_normalization_6_94718
batch_normalization_6_94720
covtr11_94723
covtr11_94725
batch_normalization_7_94729
batch_normalization_7_94731
batch_normalization_7_94733
batch_normalization_7_94735
covtr14_94738
covtr14_94740
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�covtr10/StatefulPartitionedCall�covtr11/StatefulPartitionedCall�covtr14/StatefulPartitionedCall�covtr4/StatefulPartitionedCall�covtr5/StatefulPartitionedCall�covtr6/StatefulPartitionedCall�covtr7/StatefulPartitionedCall�covtr8/StatefulPartitionedCall�covtr9/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_938082
reshape/PartitionedCall�
covtr4/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr4_94618covtr4_94620*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr4_layer_call_and_return_conditional_losses_925632 
covtr4/StatefulPartitionedCall�
leaky_re_lu/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_938262
leaky_re_lu/PartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_94624batch_normalization_94626batch_normalization_94628batch_normalization_94630*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_926662-
+batch_normalization/StatefulPartitionedCall�
covtr5/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr5_94633covtr5_94635*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr5_layer_call_and_return_conditional_losses_927152 
covtr5/StatefulPartitionedCall�
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_938792
leaky_re_lu_1/PartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_94639batch_normalization_1_94641batch_normalization_1_94643batch_normalization_1_94645*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_928182/
-batch_normalization_1/StatefulPartitionedCall�
covtr6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr6_94648covtr6_94650*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr6_layer_call_and_return_conditional_losses_928672 
covtr6/StatefulPartitionedCall�
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_939322
leaky_re_lu_2/PartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_94654batch_normalization_2_94656batch_normalization_2_94658batch_normalization_2_94660*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_929702/
-batch_normalization_2/StatefulPartitionedCall�
covtr7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0covtr7_94663covtr7_94665*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr7_layer_call_and_return_conditional_losses_930192 
covtr7/StatefulPartitionedCall�
leaky_re_lu_3/PartitionedCallPartitionedCall'covtr7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_939852
leaky_re_lu_3/PartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0batch_normalization_3_94669batch_normalization_3_94671batch_normalization_3_94673batch_normalization_3_94675*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_931222/
-batch_normalization_3/StatefulPartitionedCall�
covtr8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0covtr8_94678covtr8_94680*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr8_layer_call_and_return_conditional_losses_931712 
covtr8/StatefulPartitionedCall�
leaky_re_lu_4/PartitionedCallPartitionedCall'covtr8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_940382
leaky_re_lu_4/PartitionedCall�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0batch_normalization_4_94684batch_normalization_4_94686batch_normalization_4_94688batch_normalization_4_94690*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_932742/
-batch_normalization_4/StatefulPartitionedCall�
covtr9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0covtr9_94693covtr9_94695*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr9_layer_call_and_return_conditional_losses_933232 
covtr9/StatefulPartitionedCall�
leaky_re_lu_5/PartitionedCallPartitionedCall'covtr9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_940912
leaky_re_lu_5/PartitionedCall�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0batch_normalization_5_94699batch_normalization_5_94701batch_normalization_5_94703batch_normalization_5_94705*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_934262/
-batch_normalization_5/StatefulPartitionedCall�
covtr10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0covtr10_94708covtr10_94710*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr10_layer_call_and_return_conditional_losses_934752!
covtr10/StatefulPartitionedCall�
leaky_re_lu_6/PartitionedCallPartitionedCall(covtr10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_941442
leaky_re_lu_6/PartitionedCall�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0batch_normalization_6_94714batch_normalization_6_94716batch_normalization_6_94718batch_normalization_6_94720*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_935782/
-batch_normalization_6/StatefulPartitionedCall�
covtr11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0covtr11_94723covtr11_94725*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������"*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr11_layer_call_and_return_conditional_losses_936272!
covtr11/StatefulPartitionedCall�
leaky_re_lu_7/PartitionedCallPartitionedCall(covtr11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������"* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_941972
leaky_re_lu_7/PartitionedCall�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0batch_normalization_7_94729batch_normalization_7_94731batch_normalization_7_94733batch_normalization_7_94735*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������"*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_937302/
-batch_normalization_7/StatefulPartitionedCall�
covtr14/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0covtr14_94738covtr14_94740*
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
GPU2 *0J 8� *K
fFRD
B__inference_covtr14_layer_call_and_return_conditional_losses_937802!
covtr14/StatefulPartitionedCall�
IdentityIdentity(covtr14/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall ^covtr10/StatefulPartitionedCall ^covtr11/StatefulPartitionedCall ^covtr14/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^covtr5/StatefulPartitionedCall^covtr6/StatefulPartitionedCall^covtr7/StatefulPartitionedCall^covtr8/StatefulPartitionedCall^covtr9/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������::::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2B
covtr10/StatefulPartitionedCallcovtr10/StatefulPartitionedCall2B
covtr11/StatefulPartitionedCallcovtr11/StatefulPartitionedCall2B
covtr14/StatefulPartitionedCallcovtr14/StatefulPartitionedCall2@
covtr4/StatefulPartitionedCallcovtr4/StatefulPartitionedCall2@
covtr5/StatefulPartitionedCallcovtr5/StatefulPartitionedCall2@
covtr6/StatefulPartitionedCallcovtr6/StatefulPartitionedCall2@
covtr7/StatefulPartitionedCallcovtr7/StatefulPartitionedCall2@
covtr8/StatefulPartitionedCallcovtr8/StatefulPartitionedCall2@
covtr9/StatefulPartitionedCallcovtr9/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_93243

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
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
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_96007

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_4_layer_call_fn_96168

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
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_932432
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_7_layer_call_fn_96403

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
-:+���������������������������"*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_937302
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������"2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������"::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������"
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_95964

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_92818

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
)__inference_Generator_layer_call_fn_94612
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

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48
identity��StatefulPartitionedCall�
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
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*D
_read_only_resource_inputs&
$"	
 !"%&'(+,-.12*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_Generator_layer_call_and_return_conditional_losses_945092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
(
_output_shapes
:����������
#
_user_specified_name	gen_noise
�$
�
A__inference_covtr4_layer_call_and_return_conditional_losses_92563

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
value	B :2
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
value	B :2	
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
:*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_93091

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
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
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_95841

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
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
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
I
-__inference_leaky_re_lu_6_layer_call_fn_96265

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
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_941442
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
3__inference_batch_normalization_layer_call_fn_95885

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
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_926662
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_93985

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
)__inference_Generator_layer_call_fn_95792

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

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48
identity��StatefulPartitionedCall�
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
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_Generator_layer_call_and_return_conditional_losses_947442
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_1_layer_call_fn_95959

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
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_928182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_6_layer_call_fn_96329

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
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_935782
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_96377

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:"*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:"*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:"*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:"*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������":":":":":*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������"2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������":::::i e
A
_output_shapes/
-:+���������������������������"
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_5_layer_call_fn_96242

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
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_933952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
{
&__inference_covtr7_layer_call_fn_93029

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
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr7_layer_call_and_return_conditional_losses_930192
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
{
&__inference_covtr9_layer_call_fn_93333

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
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr9_layer_call_and_return_conditional_losses_933232
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
^
B__inference_reshape_layer_call_and_return_conditional_losses_95806

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
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
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
:���������2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_95859

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�$
�
B__inference_covtr10_layer_call_and_return_conditional_losses_93475

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
value	B :2
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
value	B :2	
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
value	B :2	
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
:*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_94144

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_94038

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_3_layer_call_fn_96094

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
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_930912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_96038

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
G
+__inference_leaky_re_lu_layer_call_fn_95821

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
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_938262
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
��
�
D__inference_Generator_layer_call_and_return_conditional_losses_94509

inputs
covtr4_94383
covtr4_94385
batch_normalization_94389
batch_normalization_94391
batch_normalization_94393
batch_normalization_94395
covtr5_94398
covtr5_94400
batch_normalization_1_94404
batch_normalization_1_94406
batch_normalization_1_94408
batch_normalization_1_94410
covtr6_94413
covtr6_94415
batch_normalization_2_94419
batch_normalization_2_94421
batch_normalization_2_94423
batch_normalization_2_94425
covtr7_94428
covtr7_94430
batch_normalization_3_94434
batch_normalization_3_94436
batch_normalization_3_94438
batch_normalization_3_94440
covtr8_94443
covtr8_94445
batch_normalization_4_94449
batch_normalization_4_94451
batch_normalization_4_94453
batch_normalization_4_94455
covtr9_94458
covtr9_94460
batch_normalization_5_94464
batch_normalization_5_94466
batch_normalization_5_94468
batch_normalization_5_94470
covtr10_94473
covtr10_94475
batch_normalization_6_94479
batch_normalization_6_94481
batch_normalization_6_94483
batch_normalization_6_94485
covtr11_94488
covtr11_94490
batch_normalization_7_94494
batch_normalization_7_94496
batch_normalization_7_94498
batch_normalization_7_94500
covtr14_94503
covtr14_94505
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�covtr10/StatefulPartitionedCall�covtr11/StatefulPartitionedCall�covtr14/StatefulPartitionedCall�covtr4/StatefulPartitionedCall�covtr5/StatefulPartitionedCall�covtr6/StatefulPartitionedCall�covtr7/StatefulPartitionedCall�covtr8/StatefulPartitionedCall�covtr9/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_938082
reshape/PartitionedCall�
covtr4/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr4_94383covtr4_94385*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr4_layer_call_and_return_conditional_losses_925632 
covtr4/StatefulPartitionedCall�
leaky_re_lu/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_938262
leaky_re_lu/PartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_94389batch_normalization_94391batch_normalization_94393batch_normalization_94395*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_926352-
+batch_normalization/StatefulPartitionedCall�
covtr5/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr5_94398covtr5_94400*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr5_layer_call_and_return_conditional_losses_927152 
covtr5/StatefulPartitionedCall�
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_938792
leaky_re_lu_1/PartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_94404batch_normalization_1_94406batch_normalization_1_94408batch_normalization_1_94410*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_927872/
-batch_normalization_1/StatefulPartitionedCall�
covtr6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr6_94413covtr6_94415*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr6_layer_call_and_return_conditional_losses_928672 
covtr6/StatefulPartitionedCall�
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_939322
leaky_re_lu_2/PartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_94419batch_normalization_2_94421batch_normalization_2_94423batch_normalization_2_94425*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_929392/
-batch_normalization_2/StatefulPartitionedCall�
covtr7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0covtr7_94428covtr7_94430*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr7_layer_call_and_return_conditional_losses_930192 
covtr7/StatefulPartitionedCall�
leaky_re_lu_3/PartitionedCallPartitionedCall'covtr7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_939852
leaky_re_lu_3/PartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0batch_normalization_3_94434batch_normalization_3_94436batch_normalization_3_94438batch_normalization_3_94440*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_930912/
-batch_normalization_3/StatefulPartitionedCall�
covtr8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0covtr8_94443covtr8_94445*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr8_layer_call_and_return_conditional_losses_931712 
covtr8/StatefulPartitionedCall�
leaky_re_lu_4/PartitionedCallPartitionedCall'covtr8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_940382
leaky_re_lu_4/PartitionedCall�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0batch_normalization_4_94449batch_normalization_4_94451batch_normalization_4_94453batch_normalization_4_94455*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_932432/
-batch_normalization_4/StatefulPartitionedCall�
covtr9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0covtr9_94458covtr9_94460*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr9_layer_call_and_return_conditional_losses_933232 
covtr9/StatefulPartitionedCall�
leaky_re_lu_5/PartitionedCallPartitionedCall'covtr9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_940912
leaky_re_lu_5/PartitionedCall�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0batch_normalization_5_94464batch_normalization_5_94466batch_normalization_5_94468batch_normalization_5_94470*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_933952/
-batch_normalization_5/StatefulPartitionedCall�
covtr10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0covtr10_94473covtr10_94475*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr10_layer_call_and_return_conditional_losses_934752!
covtr10/StatefulPartitionedCall�
leaky_re_lu_6/PartitionedCallPartitionedCall(covtr10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_941442
leaky_re_lu_6/PartitionedCall�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0batch_normalization_6_94479batch_normalization_6_94481batch_normalization_6_94483batch_normalization_6_94485*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_935472/
-batch_normalization_6/StatefulPartitionedCall�
covtr11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0covtr11_94488covtr11_94490*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������"*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr11_layer_call_and_return_conditional_losses_936272!
covtr11/StatefulPartitionedCall�
leaky_re_lu_7/PartitionedCallPartitionedCall(covtr11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������"* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_941972
leaky_re_lu_7/PartitionedCall�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0batch_normalization_7_94494batch_normalization_7_94496batch_normalization_7_94498batch_normalization_7_94500*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������"*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_936992/
-batch_normalization_7/StatefulPartitionedCall�
covtr14/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0covtr14_94503covtr14_94505*
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
GPU2 *0J 8� *K
fFRD
B__inference_covtr14_layer_call_and_return_conditional_losses_937802!
covtr14/StatefulPartitionedCall�
IdentityIdentity(covtr14/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall ^covtr10/StatefulPartitionedCall ^covtr11/StatefulPartitionedCall ^covtr14/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^covtr5/StatefulPartitionedCall^covtr6/StatefulPartitionedCall^covtr7/StatefulPartitionedCall^covtr8/StatefulPartitionedCall^covtr9/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������::::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2B
covtr10/StatefulPartitionedCallcovtr10/StatefulPartitionedCall2B
covtr11/StatefulPartitionedCallcovtr11/StatefulPartitionedCall2B
covtr14/StatefulPartitionedCallcovtr14/StatefulPartitionedCall2@
covtr4/StatefulPartitionedCallcovtr4/StatefulPartitionedCall2@
covtr5/StatefulPartitionedCallcovtr5/StatefulPartitionedCall2@
covtr6/StatefulPartitionedCallcovtr6/StatefulPartitionedCall2@
covtr7/StatefulPartitionedCallcovtr7/StatefulPartitionedCall2@
covtr8/StatefulPartitionedCallcovtr8/StatefulPartitionedCall2@
covtr9/StatefulPartitionedCallcovtr9/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_94954
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

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48
identity��StatefulPartitionedCall�
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
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������7-*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*2
config_proto" 

CPU

GPU2 *0J 8� *)
f$R"
 __inference__wrapped_model_925252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������7-2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
(
_output_shapes
:����������
#
_user_specified_name	gen_noise
�
{
&__inference_covtr6_layer_call_fn_92877

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
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr6_layer_call_and_return_conditional_losses_928672
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�$
�
B__inference_covtr11_layer_call_and_return_conditional_losses_93627

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
value	B :2
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
value	B :2	
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
value	B :"2	
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
:		"*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������"*
paddingVALID*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:"*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������"2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������"2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
��
�
D__inference_Generator_layer_call_and_return_conditional_losses_95582

inputs3
/covtr4_conv2d_transpose_readvariableop_resource*
&covtr4_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource3
/covtr5_conv2d_transpose_readvariableop_resource*
&covtr5_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource3
/covtr6_conv2d_transpose_readvariableop_resource*
&covtr6_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource3
/covtr7_conv2d_transpose_readvariableop_resource*
&covtr7_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource3
/covtr8_conv2d_transpose_readvariableop_resource*
&covtr8_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource3
/covtr9_conv2d_transpose_readvariableop_resource*
&covtr9_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource4
0covtr10_conv2d_transpose_readvariableop_resource+
'covtr10_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource4
0covtr11_conv2d_transpose_readvariableop_resource+
'covtr11_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource4
0covtr14_conv2d_transpose_readvariableop_resource+
'covtr14_biasadd_readvariableop_resource
identity�T
reshape/ShapeShapeinputs*
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
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
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
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2
reshape/Reshaped
covtr4/ShapeShapereshape/Reshape:output:0*
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
value	B :2
covtr4/stack/1b
covtr4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
covtr4/stack/2b
covtr4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
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
:*
dtype02(
&covtr4/conv2d_transpose/ReadVariableOp�
covtr4/conv2d_transposeConv2DBackpropInputcovtr4/stack:output:0.covtr4/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
covtr4/conv2d_transpose�
covtr4/BiasAdd/ReadVariableOpReadVariableOp&covtr4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr4/BiasAdd/ReadVariableOp�
covtr4/BiasAddBiasAdd covtr4/conv2d_transpose:output:0%covtr4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
covtr4/BiasAdd�
leaky_re_lu/LeakyRelu	LeakyRelucovtr4/BiasAdd:output:0*/
_output_shapes
:���������2
leaky_re_lu/LeakyRelu�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3#leaky_re_lu/LeakyRelu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2&
$batch_normalization/FusedBatchNormV3t
covtr5/ShapeShape(batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr5/Shape�
covtr5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr5/strided_slice/stack�
covtr5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr5/strided_slice/stack_1�
covtr5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr5/strided_slice/stack_2�
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
value	B :2
covtr5/stack/1b
covtr5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
covtr5/stack/2b
covtr5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr5/stack/3�
covtr5/stackPackcovtr5/strided_slice:output:0covtr5/stack/1:output:0covtr5/stack/2:output:0covtr5/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr5/stack�
covtr5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr5/strided_slice_1/stack�
covtr5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr5/strided_slice_1/stack_1�
covtr5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr5/strided_slice_1/stack_2�
covtr5/strided_slice_1StridedSlicecovtr5/stack:output:0%covtr5/strided_slice_1/stack:output:0'covtr5/strided_slice_1/stack_1:output:0'covtr5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr5/strided_slice_1�
&covtr5/conv2d_transpose/ReadVariableOpReadVariableOp/covtr5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr5/conv2d_transpose/ReadVariableOp�
covtr5/conv2d_transposeConv2DBackpropInputcovtr5/stack:output:0.covtr5/conv2d_transpose/ReadVariableOp:value:0(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
covtr5/conv2d_transpose�
covtr5/BiasAdd/ReadVariableOpReadVariableOp&covtr5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr5/BiasAdd/ReadVariableOp�
covtr5/BiasAddBiasAdd covtr5/conv2d_transpose:output:0%covtr5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
covtr5/BiasAdd�
leaky_re_lu_1/LeakyRelu	LeakyRelucovtr5/BiasAdd:output:0*/
_output_shapes
:���������2
leaky_re_lu_1/LeakyRelu�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_1/LeakyRelu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3v
covtr6/ShapeShape*batch_normalization_1/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr6/Shape�
covtr6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr6/strided_slice/stack�
covtr6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr6/strided_slice/stack_1�
covtr6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr6/strided_slice/stack_2�
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
value	B :2
covtr6/stack/1b
covtr6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
covtr6/stack/2b
covtr6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr6/stack/3�
covtr6/stackPackcovtr6/strided_slice:output:0covtr6/stack/1:output:0covtr6/stack/2:output:0covtr6/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr6/stack�
covtr6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr6/strided_slice_1/stack�
covtr6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr6/strided_slice_1/stack_1�
covtr6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr6/strided_slice_1/stack_2�
covtr6/strided_slice_1StridedSlicecovtr6/stack:output:0%covtr6/strided_slice_1/stack:output:0'covtr6/strided_slice_1/stack_1:output:0'covtr6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr6/strided_slice_1�
&covtr6/conv2d_transpose/ReadVariableOpReadVariableOp/covtr6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr6/conv2d_transpose/ReadVariableOp�
covtr6/conv2d_transposeConv2DBackpropInputcovtr6/stack:output:0.covtr6/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
covtr6/conv2d_transpose�
covtr6/BiasAdd/ReadVariableOpReadVariableOp&covtr6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr6/BiasAdd/ReadVariableOp�
covtr6/BiasAddBiasAdd covtr6/conv2d_transpose:output:0%covtr6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
covtr6/BiasAdd�
leaky_re_lu_2/LeakyRelu	LeakyRelucovtr6/BiasAdd:output:0*/
_output_shapes
:���������2
leaky_re_lu_2/LeakyRelu�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_2/LeakyRelu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3v
covtr7/ShapeShape*batch_normalization_2/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr7/Shape�
covtr7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr7/strided_slice/stack�
covtr7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr7/strided_slice/stack_1�
covtr7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr7/strided_slice/stack_2�
covtr7/strided_sliceStridedSlicecovtr7/Shape:output:0#covtr7/strided_slice/stack:output:0%covtr7/strided_slice/stack_1:output:0%covtr7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr7/strided_sliceb
covtr7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
covtr7/stack/1b
covtr7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
covtr7/stack/2b
covtr7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr7/stack/3�
covtr7/stackPackcovtr7/strided_slice:output:0covtr7/stack/1:output:0covtr7/stack/2:output:0covtr7/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr7/stack�
covtr7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr7/strided_slice_1/stack�
covtr7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr7/strided_slice_1/stack_1�
covtr7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr7/strided_slice_1/stack_2�
covtr7/strided_slice_1StridedSlicecovtr7/stack:output:0%covtr7/strided_slice_1/stack:output:0'covtr7/strided_slice_1/stack_1:output:0'covtr7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr7/strided_slice_1�
&covtr7/conv2d_transpose/ReadVariableOpReadVariableOp/covtr7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr7/conv2d_transpose/ReadVariableOp�
covtr7/conv2d_transposeConv2DBackpropInputcovtr7/stack:output:0.covtr7/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
covtr7/conv2d_transpose�
covtr7/BiasAdd/ReadVariableOpReadVariableOp&covtr7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr7/BiasAdd/ReadVariableOp�
covtr7/BiasAddBiasAdd covtr7/conv2d_transpose:output:0%covtr7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
covtr7/BiasAdd�
leaky_re_lu_3/LeakyRelu	LeakyRelucovtr7/BiasAdd:output:0*/
_output_shapes
:���������2
leaky_re_lu_3/LeakyRelu�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_3/ReadVariableOp�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_3/ReadVariableOp_1�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_3/LeakyRelu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3v
covtr8/ShapeShape*batch_normalization_3/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr8/Shape�
covtr8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr8/strided_slice/stack�
covtr8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr8/strided_slice/stack_1�
covtr8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr8/strided_slice/stack_2�
covtr8/strided_sliceStridedSlicecovtr8/Shape:output:0#covtr8/strided_slice/stack:output:0%covtr8/strided_slice/stack_1:output:0%covtr8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr8/strided_sliceb
covtr8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
covtr8/stack/1b
covtr8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
covtr8/stack/2b
covtr8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr8/stack/3�
covtr8/stackPackcovtr8/strided_slice:output:0covtr8/stack/1:output:0covtr8/stack/2:output:0covtr8/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr8/stack�
covtr8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr8/strided_slice_1/stack�
covtr8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr8/strided_slice_1/stack_1�
covtr8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr8/strided_slice_1/stack_2�
covtr8/strided_slice_1StridedSlicecovtr8/stack:output:0%covtr8/strided_slice_1/stack:output:0'covtr8/strided_slice_1/stack_1:output:0'covtr8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr8/strided_slice_1�
&covtr8/conv2d_transpose/ReadVariableOpReadVariableOp/covtr8_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr8/conv2d_transpose/ReadVariableOp�
covtr8/conv2d_transposeConv2DBackpropInputcovtr8/stack:output:0.covtr8/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
covtr8/conv2d_transpose�
covtr8/BiasAdd/ReadVariableOpReadVariableOp&covtr8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr8/BiasAdd/ReadVariableOp�
covtr8/BiasAddBiasAdd covtr8/conv2d_transpose:output:0%covtr8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
covtr8/BiasAdd�
leaky_re_lu_4/LeakyRelu	LeakyRelucovtr8/BiasAdd:output:0*/
_output_shapes
:���������2
leaky_re_lu_4/LeakyRelu�
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_4/ReadVariableOp�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_4/LeakyRelu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3v
covtr9/ShapeShape*batch_normalization_4/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr9/Shape�
covtr9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr9/strided_slice/stack�
covtr9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr9/strided_slice/stack_1�
covtr9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr9/strided_slice/stack_2�
covtr9/strided_sliceStridedSlicecovtr9/Shape:output:0#covtr9/strided_slice/stack:output:0%covtr9/strided_slice/stack_1:output:0%covtr9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr9/strided_sliceb
covtr9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :$2
covtr9/stack/1b
covtr9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
covtr9/stack/2b
covtr9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr9/stack/3�
covtr9/stackPackcovtr9/strided_slice:output:0covtr9/stack/1:output:0covtr9/stack/2:output:0covtr9/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr9/stack�
covtr9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr9/strided_slice_1/stack�
covtr9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr9/strided_slice_1/stack_1�
covtr9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr9/strided_slice_1/stack_2�
covtr9/strided_slice_1StridedSlicecovtr9/stack:output:0%covtr9/strided_slice_1/stack:output:0'covtr9/strided_slice_1/stack_1:output:0'covtr9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr9/strided_slice_1�
&covtr9/conv2d_transpose/ReadVariableOpReadVariableOp/covtr9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr9/conv2d_transpose/ReadVariableOp�
covtr9/conv2d_transposeConv2DBackpropInputcovtr9/stack:output:0.covtr9/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������$*
paddingVALID*
strides
2
covtr9/conv2d_transpose�
covtr9/BiasAdd/ReadVariableOpReadVariableOp&covtr9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr9/BiasAdd/ReadVariableOp�
covtr9/BiasAddBiasAdd covtr9/conv2d_transpose:output:0%covtr9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������$2
covtr9/BiasAdd�
leaky_re_lu_5/LeakyRelu	LeakyRelucovtr9/BiasAdd:output:0*/
_output_shapes
:���������$2
leaky_re_lu_5/LeakyRelu�
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_5/ReadVariableOp�
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_5/ReadVariableOp_1�
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_5/LeakyRelu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������$:::::*
epsilon%o�:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3x
covtr10/ShapeShape*batch_normalization_5/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr10/Shape�
covtr10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr10/strided_slice/stack�
covtr10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr10/strided_slice/stack_1�
covtr10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr10/strided_slice/stack_2�
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
value	B :+2
covtr10/stack/1d
covtr10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :!2
covtr10/stack/2d
covtr10/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr10/stack/3�
covtr10/stackPackcovtr10/strided_slice:output:0covtr10/stack/1:output:0covtr10/stack/2:output:0covtr10/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr10/stack�
covtr10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr10/strided_slice_1/stack�
covtr10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr10/strided_slice_1/stack_1�
covtr10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr10/strided_slice_1/stack_2�
covtr10/strided_slice_1StridedSlicecovtr10/stack:output:0&covtr10/strided_slice_1/stack:output:0(covtr10/strided_slice_1/stack_1:output:0(covtr10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr10/strided_slice_1�
'covtr10/conv2d_transpose/ReadVariableOpReadVariableOp0covtr10_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02)
'covtr10/conv2d_transpose/ReadVariableOp�
covtr10/conv2d_transposeConv2DBackpropInputcovtr10/stack:output:0/covtr10/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������+!*
paddingVALID*
strides
2
covtr10/conv2d_transpose�
covtr10/BiasAdd/ReadVariableOpReadVariableOp'covtr10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
covtr10/BiasAdd/ReadVariableOp�
covtr10/BiasAddBiasAdd!covtr10/conv2d_transpose:output:0&covtr10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������+!2
covtr10/BiasAdd�
leaky_re_lu_6/LeakyRelu	LeakyRelucovtr10/BiasAdd:output:0*/
_output_shapes
:���������+!2
leaky_re_lu_6/LeakyRelu�
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_6/ReadVariableOp�
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1�
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_6/LeakyRelu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������+!:::::*
epsilon%o�:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3x
covtr11/ShapeShape*batch_normalization_6/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr11/Shape�
covtr11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr11/strided_slice/stack�
covtr11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr11/strided_slice/stack_1�
covtr11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr11/strided_slice/stack_2�
covtr11/strided_sliceStridedSlicecovtr11/Shape:output:0$covtr11/strided_slice/stack:output:0&covtr11/strided_slice/stack_1:output:0&covtr11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr11/strided_sliced
covtr11/stack/1Const*
_output_shapes
: *
dtype0*
value	B :32
covtr11/stack/1d
covtr11/stack/2Const*
_output_shapes
: *
dtype0*
value	B :)2
covtr11/stack/2d
covtr11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :"2
covtr11/stack/3�
covtr11/stackPackcovtr11/strided_slice:output:0covtr11/stack/1:output:0covtr11/stack/2:output:0covtr11/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr11/stack�
covtr11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr11/strided_slice_1/stack�
covtr11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr11/strided_slice_1/stack_1�
covtr11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr11/strided_slice_1/stack_2�
covtr11/strided_slice_1StridedSlicecovtr11/stack:output:0&covtr11/strided_slice_1/stack:output:0(covtr11/strided_slice_1/stack_1:output:0(covtr11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr11/strided_slice_1�
'covtr11/conv2d_transpose/ReadVariableOpReadVariableOp0covtr11_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		"*
dtype02)
'covtr11/conv2d_transpose/ReadVariableOp�
covtr11/conv2d_transposeConv2DBackpropInputcovtr11/stack:output:0/covtr11/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������3)"*
paddingVALID*
strides
2
covtr11/conv2d_transpose�
covtr11/BiasAdd/ReadVariableOpReadVariableOp'covtr11_biasadd_readvariableop_resource*
_output_shapes
:"*
dtype02 
covtr11/BiasAdd/ReadVariableOp�
covtr11/BiasAddBiasAdd!covtr11/conv2d_transpose:output:0&covtr11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������3)"2
covtr11/BiasAdd�
leaky_re_lu_7/LeakyRelu	LeakyRelucovtr11/BiasAdd:output:0*/
_output_shapes
:���������3)"2
leaky_re_lu_7/LeakyRelu�
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:"*
dtype02&
$batch_normalization_7/ReadVariableOp�
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:"*
dtype02(
&batch_normalization_7/ReadVariableOp_1�
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:"*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:"*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_7/LeakyRelu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������3)":":":":":*
epsilon%o�:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3x
covtr14/ShapeShape*batch_normalization_7/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr14/Shape�
covtr14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr14/strided_slice/stack�
covtr14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr14/strided_slice/stack_1�
covtr14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr14/strided_slice/stack_2�
covtr14/strided_sliceStridedSlicecovtr14/Shape:output:0$covtr14/strided_slice/stack:output:0&covtr14/strided_slice/stack_1:output:0&covtr14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr14/strided_sliced
covtr14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :72
covtr14/stack/1d
covtr14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :-2
covtr14/stack/2d
covtr14/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr14/stack/3�
covtr14/stackPackcovtr14/strided_slice:output:0covtr14/stack/1:output:0covtr14/stack/2:output:0covtr14/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr14/stack�
covtr14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr14/strided_slice_1/stack�
covtr14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr14/strided_slice_1/stack_1�
covtr14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr14/strided_slice_1/stack_2�
covtr14/strided_slice_1StridedSlicecovtr14/stack:output:0&covtr14/strided_slice_1/stack:output:0(covtr14/strided_slice_1/stack_1:output:0(covtr14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr14/strided_slice_1�
'covtr14/conv2d_transpose/ReadVariableOpReadVariableOp0covtr14_conv2d_transpose_readvariableop_resource*&
_output_shapes
:"*
dtype02)
'covtr14/conv2d_transpose/ReadVariableOp�
covtr14/conv2d_transposeConv2DBackpropInputcovtr14/stack:output:0/covtr14/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������7-*
paddingVALID*
strides
2
covtr14/conv2d_transpose�
covtr14/BiasAdd/ReadVariableOpReadVariableOp'covtr14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
covtr14/BiasAdd/ReadVariableOp�
covtr14/BiasAddBiasAdd!covtr14/conv2d_transpose:output:0&covtr14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������7-2
covtr14/BiasAdd�
covtr14/SigmoidSigmoidcovtr14/BiasAdd:output:0*
T0*/
_output_shapes
:���������7-2
covtr14/Sigmoido
IdentityIdentitycovtr14/Sigmoid:y:0*
T0*/
_output_shapes
:���������7-2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������:::::::::::::::::::::::::::::::::::::::::::::::::::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_6_layer_call_fn_96316

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
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_935472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_93578

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
I
-__inference_leaky_re_lu_7_layer_call_fn_96339

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
-:+���������������������������"* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_941972
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+���������������������������"2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������":i e
A
_output_shapes/
-:+���������������������������"
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_5_layer_call_fn_96255

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
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_934262
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
b
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_95816

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
I
-__inference_leaky_re_lu_5_layer_call_fn_96191

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
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_940912
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_96303

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_2_layer_call_fn_96020

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
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_929392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_96137

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
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
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_95933

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_93699

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:"*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:"*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:"*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:"*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������":":":":":*
epsilon%o�:*
exponential_avg_factor%
�#<2
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
-:+���������������������������"2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������"::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������"
 
_user_specified_nameinputs
�$
�
A__inference_covtr9_layer_call_and_return_conditional_losses_93323

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
value	B :2
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
value	B :2	
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
value	B :2	
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
:*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
��
�
D__inference_Generator_layer_call_and_return_conditional_losses_94246
	gen_noise
covtr4_93816
covtr4_93818
batch_normalization_93860
batch_normalization_93862
batch_normalization_93864
batch_normalization_93866
covtr5_93869
covtr5_93871
batch_normalization_1_93913
batch_normalization_1_93915
batch_normalization_1_93917
batch_normalization_1_93919
covtr6_93922
covtr6_93924
batch_normalization_2_93966
batch_normalization_2_93968
batch_normalization_2_93970
batch_normalization_2_93972
covtr7_93975
covtr7_93977
batch_normalization_3_94019
batch_normalization_3_94021
batch_normalization_3_94023
batch_normalization_3_94025
covtr8_94028
covtr8_94030
batch_normalization_4_94072
batch_normalization_4_94074
batch_normalization_4_94076
batch_normalization_4_94078
covtr9_94081
covtr9_94083
batch_normalization_5_94125
batch_normalization_5_94127
batch_normalization_5_94129
batch_normalization_5_94131
covtr10_94134
covtr10_94136
batch_normalization_6_94178
batch_normalization_6_94180
batch_normalization_6_94182
batch_normalization_6_94184
covtr11_94187
covtr11_94189
batch_normalization_7_94231
batch_normalization_7_94233
batch_normalization_7_94235
batch_normalization_7_94237
covtr14_94240
covtr14_94242
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�covtr10/StatefulPartitionedCall�covtr11/StatefulPartitionedCall�covtr14/StatefulPartitionedCall�covtr4/StatefulPartitionedCall�covtr5/StatefulPartitionedCall�covtr6/StatefulPartitionedCall�covtr7/StatefulPartitionedCall�covtr8/StatefulPartitionedCall�covtr9/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCall	gen_noise*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_938082
reshape/PartitionedCall�
covtr4/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr4_93816covtr4_93818*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr4_layer_call_and_return_conditional_losses_925632 
covtr4/StatefulPartitionedCall�
leaky_re_lu/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_938262
leaky_re_lu/PartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_93860batch_normalization_93862batch_normalization_93864batch_normalization_93866*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_926352-
+batch_normalization/StatefulPartitionedCall�
covtr5/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr5_93869covtr5_93871*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr5_layer_call_and_return_conditional_losses_927152 
covtr5/StatefulPartitionedCall�
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_938792
leaky_re_lu_1/PartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_93913batch_normalization_1_93915batch_normalization_1_93917batch_normalization_1_93919*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_927872/
-batch_normalization_1/StatefulPartitionedCall�
covtr6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr6_93922covtr6_93924*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr6_layer_call_and_return_conditional_losses_928672 
covtr6/StatefulPartitionedCall�
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_939322
leaky_re_lu_2/PartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_93966batch_normalization_2_93968batch_normalization_2_93970batch_normalization_2_93972*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_929392/
-batch_normalization_2/StatefulPartitionedCall�
covtr7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0covtr7_93975covtr7_93977*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr7_layer_call_and_return_conditional_losses_930192 
covtr7/StatefulPartitionedCall�
leaky_re_lu_3/PartitionedCallPartitionedCall'covtr7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_939852
leaky_re_lu_3/PartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0batch_normalization_3_94019batch_normalization_3_94021batch_normalization_3_94023batch_normalization_3_94025*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_930912/
-batch_normalization_3/StatefulPartitionedCall�
covtr8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0covtr8_94028covtr8_94030*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr8_layer_call_and_return_conditional_losses_931712 
covtr8/StatefulPartitionedCall�
leaky_re_lu_4/PartitionedCallPartitionedCall'covtr8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_940382
leaky_re_lu_4/PartitionedCall�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0batch_normalization_4_94072batch_normalization_4_94074batch_normalization_4_94076batch_normalization_4_94078*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_932432/
-batch_normalization_4/StatefulPartitionedCall�
covtr9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0covtr9_94081covtr9_94083*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr9_layer_call_and_return_conditional_losses_933232 
covtr9/StatefulPartitionedCall�
leaky_re_lu_5/PartitionedCallPartitionedCall'covtr9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_940912
leaky_re_lu_5/PartitionedCall�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0batch_normalization_5_94125batch_normalization_5_94127batch_normalization_5_94129batch_normalization_5_94131*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_933952/
-batch_normalization_5/StatefulPartitionedCall�
covtr10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0covtr10_94134covtr10_94136*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr10_layer_call_and_return_conditional_losses_934752!
covtr10/StatefulPartitionedCall�
leaky_re_lu_6/PartitionedCallPartitionedCall(covtr10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_941442
leaky_re_lu_6/PartitionedCall�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0batch_normalization_6_94178batch_normalization_6_94180batch_normalization_6_94182batch_normalization_6_94184*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_935472/
-batch_normalization_6/StatefulPartitionedCall�
covtr11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0covtr11_94187covtr11_94189*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������"*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr11_layer_call_and_return_conditional_losses_936272!
covtr11/StatefulPartitionedCall�
leaky_re_lu_7/PartitionedCallPartitionedCall(covtr11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������"* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_941972
leaky_re_lu_7/PartitionedCall�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0batch_normalization_7_94231batch_normalization_7_94233batch_normalization_7_94235batch_normalization_7_94237*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������"*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_936992/
-batch_normalization_7/StatefulPartitionedCall�
covtr14/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0covtr14_94240covtr14_94242*
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
GPU2 *0J 8� *K
fFRD
B__inference_covtr14_layer_call_and_return_conditional_losses_937802!
covtr14/StatefulPartitionedCall�
IdentityIdentity(covtr14/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall ^covtr10/StatefulPartitionedCall ^covtr11/StatefulPartitionedCall ^covtr14/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^covtr5/StatefulPartitionedCall^covtr6/StatefulPartitionedCall^covtr7/StatefulPartitionedCall^covtr8/StatefulPartitionedCall^covtr9/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������::::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2B
covtr10/StatefulPartitionedCallcovtr10/StatefulPartitionedCall2B
covtr11/StatefulPartitionedCallcovtr11/StatefulPartitionedCall2B
covtr14/StatefulPartitionedCallcovtr14/StatefulPartitionedCall2@
covtr4/StatefulPartitionedCallcovtr4/StatefulPartitionedCall2@
covtr5/StatefulPartitionedCallcovtr5/StatefulPartitionedCall2@
covtr6/StatefulPartitionedCallcovtr6/StatefulPartitionedCall2@
covtr7/StatefulPartitionedCallcovtr7/StatefulPartitionedCall2@
covtr8/StatefulPartitionedCallcovtr8/StatefulPartitionedCall2@
covtr9/StatefulPartitionedCallcovtr9/StatefulPartitionedCall:S O
(
_output_shapes
:����������
#
_user_specified_name	gen_noise
�
I
-__inference_leaky_re_lu_4_layer_call_fn_96117

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
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_940382
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_92970

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_3_layer_call_fn_96107

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
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_931222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
Ʌ
�
 __inference__wrapped_model_92525
	gen_noise=
9generator_covtr4_conv2d_transpose_readvariableop_resource4
0generator_covtr4_biasadd_readvariableop_resource9
5generator_batch_normalization_readvariableop_resource;
7generator_batch_normalization_readvariableop_1_resourceJ
Fgenerator_batch_normalization_fusedbatchnormv3_readvariableop_resourceL
Hgenerator_batch_normalization_fusedbatchnormv3_readvariableop_1_resource=
9generator_covtr5_conv2d_transpose_readvariableop_resource4
0generator_covtr5_biasadd_readvariableop_resource;
7generator_batch_normalization_1_readvariableop_resource=
9generator_batch_normalization_1_readvariableop_1_resourceL
Hgenerator_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceN
Jgenerator_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource=
9generator_covtr6_conv2d_transpose_readvariableop_resource4
0generator_covtr6_biasadd_readvariableop_resource;
7generator_batch_normalization_2_readvariableop_resource=
9generator_batch_normalization_2_readvariableop_1_resourceL
Hgenerator_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceN
Jgenerator_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource=
9generator_covtr7_conv2d_transpose_readvariableop_resource4
0generator_covtr7_biasadd_readvariableop_resource;
7generator_batch_normalization_3_readvariableop_resource=
9generator_batch_normalization_3_readvariableop_1_resourceL
Hgenerator_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceN
Jgenerator_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource=
9generator_covtr8_conv2d_transpose_readvariableop_resource4
0generator_covtr8_biasadd_readvariableop_resource;
7generator_batch_normalization_4_readvariableop_resource=
9generator_batch_normalization_4_readvariableop_1_resourceL
Hgenerator_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceN
Jgenerator_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource=
9generator_covtr9_conv2d_transpose_readvariableop_resource4
0generator_covtr9_biasadd_readvariableop_resource;
7generator_batch_normalization_5_readvariableop_resource=
9generator_batch_normalization_5_readvariableop_1_resourceL
Hgenerator_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceN
Jgenerator_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource>
:generator_covtr10_conv2d_transpose_readvariableop_resource5
1generator_covtr10_biasadd_readvariableop_resource;
7generator_batch_normalization_6_readvariableop_resource=
9generator_batch_normalization_6_readvariableop_1_resourceL
Hgenerator_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceN
Jgenerator_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource>
:generator_covtr11_conv2d_transpose_readvariableop_resource5
1generator_covtr11_biasadd_readvariableop_resource;
7generator_batch_normalization_7_readvariableop_resource=
9generator_batch_normalization_7_readvariableop_1_resourceL
Hgenerator_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceN
Jgenerator_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource>
:generator_covtr14_conv2d_transpose_readvariableop_resource5
1generator_covtr14_biasadd_readvariableop_resource
identity�k
Generator/reshape/ShapeShape	gen_noise*
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
value	B :2#
!Generator/reshape/Reshape/shape/1�
!Generator/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
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
Generator/reshape/ReshapeReshape	gen_noise(Generator/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2
Generator/reshape/Reshape�
Generator/covtr4/ShapeShape"Generator/reshape/Reshape:output:0*
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
value	B :2
Generator/covtr4/stack/1v
Generator/covtr4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr4/stack/2v
Generator/covtr4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
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
:*
dtype022
0Generator/covtr4/conv2d_transpose/ReadVariableOp�
!Generator/covtr4/conv2d_transposeConv2DBackpropInputGenerator/covtr4/stack:output:08Generator/covtr4/conv2d_transpose/ReadVariableOp:value:0"Generator/reshape/Reshape:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2#
!Generator/covtr4/conv2d_transpose�
'Generator/covtr4/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Generator/covtr4/BiasAdd/ReadVariableOp�
Generator/covtr4/BiasAddBiasAdd*Generator/covtr4/conv2d_transpose:output:0/Generator/covtr4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
Generator/covtr4/BiasAdd�
Generator/leaky_re_lu/LeakyRelu	LeakyRelu!Generator/covtr4/BiasAdd:output:0*/
_output_shapes
:���������2!
Generator/leaky_re_lu/LeakyRelu�
,Generator/batch_normalization/ReadVariableOpReadVariableOp5generator_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02.
,Generator/batch_normalization/ReadVariableOp�
.Generator/batch_normalization/ReadVariableOp_1ReadVariableOp7generator_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype020
.Generator/batch_normalization/ReadVariableOp_1�
=Generator/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpFgenerator_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=Generator/batch_normalization/FusedBatchNormV3/ReadVariableOp�
?Generator/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHgenerator_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?Generator/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
.Generator/batch_normalization/FusedBatchNormV3FusedBatchNormV3-Generator/leaky_re_lu/LeakyRelu:activations:04Generator/batch_normalization/ReadVariableOp:value:06Generator/batch_normalization/ReadVariableOp_1:value:0EGenerator/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0GGenerator/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 20
.Generator/batch_normalization/FusedBatchNormV3�
Generator/covtr5/ShapeShape2Generator/batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr5/Shape�
$Generator/covtr5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr5/strided_slice/stack�
&Generator/covtr5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr5/strided_slice/stack_1�
&Generator/covtr5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr5/strided_slice/stack_2�
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
value	B :2
Generator/covtr5/stack/1v
Generator/covtr5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr5/stack/2v
Generator/covtr5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr5/stack/3�
Generator/covtr5/stackPack'Generator/covtr5/strided_slice:output:0!Generator/covtr5/stack/1:output:0!Generator/covtr5/stack/2:output:0!Generator/covtr5/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr5/stack�
&Generator/covtr5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr5/strided_slice_1/stack�
(Generator/covtr5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr5/strided_slice_1/stack_1�
(Generator/covtr5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr5/strided_slice_1/stack_2�
 Generator/covtr5/strided_slice_1StridedSliceGenerator/covtr5/stack:output:0/Generator/covtr5/strided_slice_1/stack:output:01Generator/covtr5/strided_slice_1/stack_1:output:01Generator/covtr5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr5/strided_slice_1�
0Generator/covtr5/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0Generator/covtr5/conv2d_transpose/ReadVariableOp�
!Generator/covtr5/conv2d_transposeConv2DBackpropInputGenerator/covtr5/stack:output:08Generator/covtr5/conv2d_transpose/ReadVariableOp:value:02Generator/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2#
!Generator/covtr5/conv2d_transpose�
'Generator/covtr5/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Generator/covtr5/BiasAdd/ReadVariableOp�
Generator/covtr5/BiasAddBiasAdd*Generator/covtr5/conv2d_transpose:output:0/Generator/covtr5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
Generator/covtr5/BiasAdd�
!Generator/leaky_re_lu_1/LeakyRelu	LeakyRelu!Generator/covtr5/BiasAdd:output:0*/
_output_shapes
:���������2#
!Generator/leaky_re_lu_1/LeakyRelu�
.Generator/batch_normalization_1/ReadVariableOpReadVariableOp7generator_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype020
.Generator/batch_normalization_1/ReadVariableOp�
0Generator/batch_normalization_1/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype022
0Generator/batch_normalization_1/ReadVariableOp_1�
?Generator/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?Generator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
AGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
AGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
0Generator/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_1/LeakyRelu:activations:06Generator/batch_normalization_1/ReadVariableOp:value:08Generator/batch_normalization_1/ReadVariableOp_1:value:0GGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 22
0Generator/batch_normalization_1/FusedBatchNormV3�
Generator/covtr6/ShapeShape4Generator/batch_normalization_1/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr6/Shape�
$Generator/covtr6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr6/strided_slice/stack�
&Generator/covtr6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr6/strided_slice/stack_1�
&Generator/covtr6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr6/strided_slice/stack_2�
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
value	B :2
Generator/covtr6/stack/1v
Generator/covtr6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr6/stack/2v
Generator/covtr6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr6/stack/3�
Generator/covtr6/stackPack'Generator/covtr6/strided_slice:output:0!Generator/covtr6/stack/1:output:0!Generator/covtr6/stack/2:output:0!Generator/covtr6/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr6/stack�
&Generator/covtr6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr6/strided_slice_1/stack�
(Generator/covtr6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr6/strided_slice_1/stack_1�
(Generator/covtr6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr6/strided_slice_1/stack_2�
 Generator/covtr6/strided_slice_1StridedSliceGenerator/covtr6/stack:output:0/Generator/covtr6/strided_slice_1/stack:output:01Generator/covtr6/strided_slice_1/stack_1:output:01Generator/covtr6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr6/strided_slice_1�
0Generator/covtr6/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0Generator/covtr6/conv2d_transpose/ReadVariableOp�
!Generator/covtr6/conv2d_transposeConv2DBackpropInputGenerator/covtr6/stack:output:08Generator/covtr6/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2#
!Generator/covtr6/conv2d_transpose�
'Generator/covtr6/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Generator/covtr6/BiasAdd/ReadVariableOp�
Generator/covtr6/BiasAddBiasAdd*Generator/covtr6/conv2d_transpose:output:0/Generator/covtr6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
Generator/covtr6/BiasAdd�
!Generator/leaky_re_lu_2/LeakyRelu	LeakyRelu!Generator/covtr6/BiasAdd:output:0*/
_output_shapes
:���������2#
!Generator/leaky_re_lu_2/LeakyRelu�
.Generator/batch_normalization_2/ReadVariableOpReadVariableOp7generator_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype020
.Generator/batch_normalization_2/ReadVariableOp�
0Generator/batch_normalization_2/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype022
0Generator/batch_normalization_2/ReadVariableOp_1�
?Generator/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?Generator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
AGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
AGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
0Generator/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_2/LeakyRelu:activations:06Generator/batch_normalization_2/ReadVariableOp:value:08Generator/batch_normalization_2/ReadVariableOp_1:value:0GGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 22
0Generator/batch_normalization_2/FusedBatchNormV3�
Generator/covtr7/ShapeShape4Generator/batch_normalization_2/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr7/Shape�
$Generator/covtr7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr7/strided_slice/stack�
&Generator/covtr7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr7/strided_slice/stack_1�
&Generator/covtr7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr7/strided_slice/stack_2�
Generator/covtr7/strided_sliceStridedSliceGenerator/covtr7/Shape:output:0-Generator/covtr7/strided_slice/stack:output:0/Generator/covtr7/strided_slice/stack_1:output:0/Generator/covtr7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Generator/covtr7/strided_slicev
Generator/covtr7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr7/stack/1v
Generator/covtr7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr7/stack/2v
Generator/covtr7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr7/stack/3�
Generator/covtr7/stackPack'Generator/covtr7/strided_slice:output:0!Generator/covtr7/stack/1:output:0!Generator/covtr7/stack/2:output:0!Generator/covtr7/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr7/stack�
&Generator/covtr7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr7/strided_slice_1/stack�
(Generator/covtr7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr7/strided_slice_1/stack_1�
(Generator/covtr7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr7/strided_slice_1/stack_2�
 Generator/covtr7/strided_slice_1StridedSliceGenerator/covtr7/stack:output:0/Generator/covtr7/strided_slice_1/stack:output:01Generator/covtr7/strided_slice_1/stack_1:output:01Generator/covtr7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr7/strided_slice_1�
0Generator/covtr7/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0Generator/covtr7/conv2d_transpose/ReadVariableOp�
!Generator/covtr7/conv2d_transposeConv2DBackpropInputGenerator/covtr7/stack:output:08Generator/covtr7/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2#
!Generator/covtr7/conv2d_transpose�
'Generator/covtr7/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Generator/covtr7/BiasAdd/ReadVariableOp�
Generator/covtr7/BiasAddBiasAdd*Generator/covtr7/conv2d_transpose:output:0/Generator/covtr7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
Generator/covtr7/BiasAdd�
!Generator/leaky_re_lu_3/LeakyRelu	LeakyRelu!Generator/covtr7/BiasAdd:output:0*/
_output_shapes
:���������2#
!Generator/leaky_re_lu_3/LeakyRelu�
.Generator/batch_normalization_3/ReadVariableOpReadVariableOp7generator_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype020
.Generator/batch_normalization_3/ReadVariableOp�
0Generator/batch_normalization_3/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype022
0Generator/batch_normalization_3/ReadVariableOp_1�
?Generator/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?Generator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
AGenerator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
AGenerator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
0Generator/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_3/LeakyRelu:activations:06Generator/batch_normalization_3/ReadVariableOp:value:08Generator/batch_normalization_3/ReadVariableOp_1:value:0GGenerator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 22
0Generator/batch_normalization_3/FusedBatchNormV3�
Generator/covtr8/ShapeShape4Generator/batch_normalization_3/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr8/Shape�
$Generator/covtr8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr8/strided_slice/stack�
&Generator/covtr8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr8/strided_slice/stack_1�
&Generator/covtr8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr8/strided_slice/stack_2�
Generator/covtr8/strided_sliceStridedSliceGenerator/covtr8/Shape:output:0-Generator/covtr8/strided_slice/stack:output:0/Generator/covtr8/strided_slice/stack_1:output:0/Generator/covtr8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Generator/covtr8/strided_slicev
Generator/covtr8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr8/stack/1v
Generator/covtr8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr8/stack/2v
Generator/covtr8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr8/stack/3�
Generator/covtr8/stackPack'Generator/covtr8/strided_slice:output:0!Generator/covtr8/stack/1:output:0!Generator/covtr8/stack/2:output:0!Generator/covtr8/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr8/stack�
&Generator/covtr8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr8/strided_slice_1/stack�
(Generator/covtr8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr8/strided_slice_1/stack_1�
(Generator/covtr8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr8/strided_slice_1/stack_2�
 Generator/covtr8/strided_slice_1StridedSliceGenerator/covtr8/stack:output:0/Generator/covtr8/strided_slice_1/stack:output:01Generator/covtr8/strided_slice_1/stack_1:output:01Generator/covtr8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr8/strided_slice_1�
0Generator/covtr8/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr8_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0Generator/covtr8/conv2d_transpose/ReadVariableOp�
!Generator/covtr8/conv2d_transposeConv2DBackpropInputGenerator/covtr8/stack:output:08Generator/covtr8/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2#
!Generator/covtr8/conv2d_transpose�
'Generator/covtr8/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Generator/covtr8/BiasAdd/ReadVariableOp�
Generator/covtr8/BiasAddBiasAdd*Generator/covtr8/conv2d_transpose:output:0/Generator/covtr8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
Generator/covtr8/BiasAdd�
!Generator/leaky_re_lu_4/LeakyRelu	LeakyRelu!Generator/covtr8/BiasAdd:output:0*/
_output_shapes
:���������2#
!Generator/leaky_re_lu_4/LeakyRelu�
.Generator/batch_normalization_4/ReadVariableOpReadVariableOp7generator_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype020
.Generator/batch_normalization_4/ReadVariableOp�
0Generator/batch_normalization_4/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype022
0Generator/batch_normalization_4/ReadVariableOp_1�
?Generator/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?Generator/batch_normalization_4/FusedBatchNormV3/ReadVariableOp�
AGenerator/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
AGenerator/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�
0Generator/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_4/LeakyRelu:activations:06Generator/batch_normalization_4/ReadVariableOp:value:08Generator/batch_normalization_4/ReadVariableOp_1:value:0GGenerator/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 22
0Generator/batch_normalization_4/FusedBatchNormV3�
Generator/covtr9/ShapeShape4Generator/batch_normalization_4/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr9/Shape�
$Generator/covtr9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr9/strided_slice/stack�
&Generator/covtr9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr9/strided_slice/stack_1�
&Generator/covtr9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr9/strided_slice/stack_2�
Generator/covtr9/strided_sliceStridedSliceGenerator/covtr9/Shape:output:0-Generator/covtr9/strided_slice/stack:output:0/Generator/covtr9/strided_slice/stack_1:output:0/Generator/covtr9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Generator/covtr9/strided_slicev
Generator/covtr9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :$2
Generator/covtr9/stack/1v
Generator/covtr9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr9/stack/2v
Generator/covtr9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr9/stack/3�
Generator/covtr9/stackPack'Generator/covtr9/strided_slice:output:0!Generator/covtr9/stack/1:output:0!Generator/covtr9/stack/2:output:0!Generator/covtr9/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr9/stack�
&Generator/covtr9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr9/strided_slice_1/stack�
(Generator/covtr9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr9/strided_slice_1/stack_1�
(Generator/covtr9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr9/strided_slice_1/stack_2�
 Generator/covtr9/strided_slice_1StridedSliceGenerator/covtr9/stack:output:0/Generator/covtr9/strided_slice_1/stack:output:01Generator/covtr9/strided_slice_1/stack_1:output:01Generator/covtr9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr9/strided_slice_1�
0Generator/covtr9/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0Generator/covtr9/conv2d_transpose/ReadVariableOp�
!Generator/covtr9/conv2d_transposeConv2DBackpropInputGenerator/covtr9/stack:output:08Generator/covtr9/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������$*
paddingVALID*
strides
2#
!Generator/covtr9/conv2d_transpose�
'Generator/covtr9/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Generator/covtr9/BiasAdd/ReadVariableOp�
Generator/covtr9/BiasAddBiasAdd*Generator/covtr9/conv2d_transpose:output:0/Generator/covtr9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������$2
Generator/covtr9/BiasAdd�
!Generator/leaky_re_lu_5/LeakyRelu	LeakyRelu!Generator/covtr9/BiasAdd:output:0*/
_output_shapes
:���������$2#
!Generator/leaky_re_lu_5/LeakyRelu�
.Generator/batch_normalization_5/ReadVariableOpReadVariableOp7generator_batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype020
.Generator/batch_normalization_5/ReadVariableOp�
0Generator/batch_normalization_5/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype022
0Generator/batch_normalization_5/ReadVariableOp_1�
?Generator/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?Generator/batch_normalization_5/FusedBatchNormV3/ReadVariableOp�
AGenerator/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
AGenerator/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�
0Generator/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_5/LeakyRelu:activations:06Generator/batch_normalization_5/ReadVariableOp:value:08Generator/batch_normalization_5/ReadVariableOp_1:value:0GGenerator/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������$:::::*
epsilon%o�:*
is_training( 22
0Generator/batch_normalization_5/FusedBatchNormV3�
Generator/covtr10/ShapeShape4Generator/batch_normalization_5/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr10/Shape�
%Generator/covtr10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Generator/covtr10/strided_slice/stack�
'Generator/covtr10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/covtr10/strided_slice/stack_1�
'Generator/covtr10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/covtr10/strided_slice/stack_2�
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
value	B :+2
Generator/covtr10/stack/1x
Generator/covtr10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :!2
Generator/covtr10/stack/2x
Generator/covtr10/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr10/stack/3�
Generator/covtr10/stackPack(Generator/covtr10/strided_slice:output:0"Generator/covtr10/stack/1:output:0"Generator/covtr10/stack/2:output:0"Generator/covtr10/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr10/stack�
'Generator/covtr10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'Generator/covtr10/strided_slice_1/stack�
)Generator/covtr10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)Generator/covtr10/strided_slice_1/stack_1�
)Generator/covtr10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)Generator/covtr10/strided_slice_1/stack_2�
!Generator/covtr10/strided_slice_1StridedSlice Generator/covtr10/stack:output:00Generator/covtr10/strided_slice_1/stack:output:02Generator/covtr10/strided_slice_1/stack_1:output:02Generator/covtr10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!Generator/covtr10/strided_slice_1�
1Generator/covtr10/conv2d_transpose/ReadVariableOpReadVariableOp:generator_covtr10_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype023
1Generator/covtr10/conv2d_transpose/ReadVariableOp�
"Generator/covtr10/conv2d_transposeConv2DBackpropInput Generator/covtr10/stack:output:09Generator/covtr10/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������+!*
paddingVALID*
strides
2$
"Generator/covtr10/conv2d_transpose�
(Generator/covtr10/BiasAdd/ReadVariableOpReadVariableOp1generator_covtr10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Generator/covtr10/BiasAdd/ReadVariableOp�
Generator/covtr10/BiasAddBiasAdd+Generator/covtr10/conv2d_transpose:output:00Generator/covtr10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������+!2
Generator/covtr10/BiasAdd�
!Generator/leaky_re_lu_6/LeakyRelu	LeakyRelu"Generator/covtr10/BiasAdd:output:0*/
_output_shapes
:���������+!2#
!Generator/leaky_re_lu_6/LeakyRelu�
.Generator/batch_normalization_6/ReadVariableOpReadVariableOp7generator_batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype020
.Generator/batch_normalization_6/ReadVariableOp�
0Generator/batch_normalization_6/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype022
0Generator/batch_normalization_6/ReadVariableOp_1�
?Generator/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?Generator/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�
AGenerator/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
AGenerator/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�
0Generator/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_6/LeakyRelu:activations:06Generator/batch_normalization_6/ReadVariableOp:value:08Generator/batch_normalization_6/ReadVariableOp_1:value:0GGenerator/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������+!:::::*
epsilon%o�:*
is_training( 22
0Generator/batch_normalization_6/FusedBatchNormV3�
Generator/covtr11/ShapeShape4Generator/batch_normalization_6/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr11/Shape�
%Generator/covtr11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Generator/covtr11/strided_slice/stack�
'Generator/covtr11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/covtr11/strided_slice/stack_1�
'Generator/covtr11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/covtr11/strided_slice/stack_2�
Generator/covtr11/strided_sliceStridedSlice Generator/covtr11/Shape:output:0.Generator/covtr11/strided_slice/stack:output:00Generator/covtr11/strided_slice/stack_1:output:00Generator/covtr11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
Generator/covtr11/strided_slicex
Generator/covtr11/stack/1Const*
_output_shapes
: *
dtype0*
value	B :32
Generator/covtr11/stack/1x
Generator/covtr11/stack/2Const*
_output_shapes
: *
dtype0*
value	B :)2
Generator/covtr11/stack/2x
Generator/covtr11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :"2
Generator/covtr11/stack/3�
Generator/covtr11/stackPack(Generator/covtr11/strided_slice:output:0"Generator/covtr11/stack/1:output:0"Generator/covtr11/stack/2:output:0"Generator/covtr11/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr11/stack�
'Generator/covtr11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'Generator/covtr11/strided_slice_1/stack�
)Generator/covtr11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)Generator/covtr11/strided_slice_1/stack_1�
)Generator/covtr11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)Generator/covtr11/strided_slice_1/stack_2�
!Generator/covtr11/strided_slice_1StridedSlice Generator/covtr11/stack:output:00Generator/covtr11/strided_slice_1/stack:output:02Generator/covtr11/strided_slice_1/stack_1:output:02Generator/covtr11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!Generator/covtr11/strided_slice_1�
1Generator/covtr11/conv2d_transpose/ReadVariableOpReadVariableOp:generator_covtr11_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		"*
dtype023
1Generator/covtr11/conv2d_transpose/ReadVariableOp�
"Generator/covtr11/conv2d_transposeConv2DBackpropInput Generator/covtr11/stack:output:09Generator/covtr11/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������3)"*
paddingVALID*
strides
2$
"Generator/covtr11/conv2d_transpose�
(Generator/covtr11/BiasAdd/ReadVariableOpReadVariableOp1generator_covtr11_biasadd_readvariableop_resource*
_output_shapes
:"*
dtype02*
(Generator/covtr11/BiasAdd/ReadVariableOp�
Generator/covtr11/BiasAddBiasAdd+Generator/covtr11/conv2d_transpose:output:00Generator/covtr11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������3)"2
Generator/covtr11/BiasAdd�
!Generator/leaky_re_lu_7/LeakyRelu	LeakyRelu"Generator/covtr11/BiasAdd:output:0*/
_output_shapes
:���������3)"2#
!Generator/leaky_re_lu_7/LeakyRelu�
.Generator/batch_normalization_7/ReadVariableOpReadVariableOp7generator_batch_normalization_7_readvariableop_resource*
_output_shapes
:"*
dtype020
.Generator/batch_normalization_7/ReadVariableOp�
0Generator/batch_normalization_7/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:"*
dtype022
0Generator/batch_normalization_7/ReadVariableOp_1�
?Generator/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:"*
dtype02A
?Generator/batch_normalization_7/FusedBatchNormV3/ReadVariableOp�
AGenerator/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:"*
dtype02C
AGenerator/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�
0Generator/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_7/LeakyRelu:activations:06Generator/batch_normalization_7/ReadVariableOp:value:08Generator/batch_normalization_7/ReadVariableOp_1:value:0GGenerator/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������3)":":":":":*
epsilon%o�:*
is_training( 22
0Generator/batch_normalization_7/FusedBatchNormV3�
Generator/covtr14/ShapeShape4Generator/batch_normalization_7/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr14/Shape�
%Generator/covtr14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Generator/covtr14/strided_slice/stack�
'Generator/covtr14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/covtr14/strided_slice/stack_1�
'Generator/covtr14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/covtr14/strided_slice/stack_2�
Generator/covtr14/strided_sliceStridedSlice Generator/covtr14/Shape:output:0.Generator/covtr14/strided_slice/stack:output:00Generator/covtr14/strided_slice/stack_1:output:00Generator/covtr14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
Generator/covtr14/strided_slicex
Generator/covtr14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :72
Generator/covtr14/stack/1x
Generator/covtr14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :-2
Generator/covtr14/stack/2x
Generator/covtr14/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Generator/covtr14/stack/3�
Generator/covtr14/stackPack(Generator/covtr14/strided_slice:output:0"Generator/covtr14/stack/1:output:0"Generator/covtr14/stack/2:output:0"Generator/covtr14/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr14/stack�
'Generator/covtr14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'Generator/covtr14/strided_slice_1/stack�
)Generator/covtr14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)Generator/covtr14/strided_slice_1/stack_1�
)Generator/covtr14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)Generator/covtr14/strided_slice_1/stack_2�
!Generator/covtr14/strided_slice_1StridedSlice Generator/covtr14/stack:output:00Generator/covtr14/strided_slice_1/stack:output:02Generator/covtr14/strided_slice_1/stack_1:output:02Generator/covtr14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!Generator/covtr14/strided_slice_1�
1Generator/covtr14/conv2d_transpose/ReadVariableOpReadVariableOp:generator_covtr14_conv2d_transpose_readvariableop_resource*&
_output_shapes
:"*
dtype023
1Generator/covtr14/conv2d_transpose/ReadVariableOp�
"Generator/covtr14/conv2d_transposeConv2DBackpropInput Generator/covtr14/stack:output:09Generator/covtr14/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������7-*
paddingVALID*
strides
2$
"Generator/covtr14/conv2d_transpose�
(Generator/covtr14/BiasAdd/ReadVariableOpReadVariableOp1generator_covtr14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Generator/covtr14/BiasAdd/ReadVariableOp�
Generator/covtr14/BiasAddBiasAdd+Generator/covtr14/conv2d_transpose:output:00Generator/covtr14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������7-2
Generator/covtr14/BiasAdd�
Generator/covtr14/SigmoidSigmoid"Generator/covtr14/BiasAdd:output:0*
T0*/
_output_shapes
:���������7-2
Generator/covtr14/Sigmoidy
IdentityIdentityGenerator/covtr14/Sigmoid:y:0*
T0*/
_output_shapes
:���������7-2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������:::::::::::::::::::::::::::::::::::::::::::::::::::S O
(
_output_shapes
:����������
#
_user_specified_name	gen_noise
�
�
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_93395

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
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
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_95890

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_93730

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:"*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:"*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:"*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:"*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������":":":":":*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������"2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������":::::i e
A
_output_shapes/
-:+���������������������������"
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_95989

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
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
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�$
�
A__inference_covtr5_layer_call_and_return_conditional_losses_92715

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
value	B :2	
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
value	B :2	
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
:*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
C
'__inference_reshape_layer_call_fn_95811

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
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_938082
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_96112

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_4_layer_call_fn_96181

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
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_932742
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_94091

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_96260

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�%
�
B__inference_covtr14_layer_call_and_return_conditional_losses_93780

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
value	B :2
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
value	B :2	
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
:"*
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
5:+���������������������������":::i e
A
_output_shapes/
-:+���������������������������"
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_2_layer_call_fn_96033

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
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_929702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_7_layer_call_fn_96390

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
-:+���������������������������"*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_936992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������"2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������"::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������"
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_96334

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������"2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������"2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������":i e
A
_output_shapes/
-:+���������������������������"
 
_user_specified_nameinputs
�$
�
A__inference_covtr7_layer_call_and_return_conditional_losses_93019

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
value	B :2
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
value	B :2	
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
value	B :2	
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
:*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
{
&__inference_covtr8_layer_call_fn_93181

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
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_covtr8_layer_call_and_return_conditional_losses_931712
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_95915

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
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
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�$
�
A__inference_covtr8_layer_call_and_return_conditional_losses_93171

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
value	B :2	
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
value	B :2	
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
:*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
)__inference_Generator_layer_call_fn_94847
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

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48
identity��StatefulPartitionedCall�
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
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_Generator_layer_call_and_return_conditional_losses_947442
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
(
_output_shapes
:����������
#
_user_specified_name	gen_noise
�
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_96081

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
I
-__inference_leaky_re_lu_3_layer_call_fn_96043

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
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_939852
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_93122

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
|
'__inference_covtr11_layer_call_fn_93637

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
-:+���������������������������"*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_covtr11_layer_call_and_return_conditional_losses_936272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������"2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_93879

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_92939

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
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
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
ԑ
�
D__inference_Generator_layer_call_and_return_conditional_losses_95276

inputs3
/covtr4_conv2d_transpose_readvariableop_resource*
&covtr4_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource3
/covtr5_conv2d_transpose_readvariableop_resource*
&covtr5_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource3
/covtr6_conv2d_transpose_readvariableop_resource*
&covtr6_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource3
/covtr7_conv2d_transpose_readvariableop_resource*
&covtr7_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource3
/covtr8_conv2d_transpose_readvariableop_resource*
&covtr8_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource3
/covtr9_conv2d_transpose_readvariableop_resource*
&covtr9_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource4
0covtr10_conv2d_transpose_readvariableop_resource+
'covtr10_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource4
0covtr11_conv2d_transpose_readvariableop_resource+
'covtr11_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource4
0covtr14_conv2d_transpose_readvariableop_resource+
'covtr14_biasadd_readvariableop_resource
identity��"batch_normalization/AssignNewValue�$batch_normalization/AssignNewValue_1�$batch_normalization_1/AssignNewValue�&batch_normalization_1/AssignNewValue_1�$batch_normalization_2/AssignNewValue�&batch_normalization_2/AssignNewValue_1�$batch_normalization_3/AssignNewValue�&batch_normalization_3/AssignNewValue_1�$batch_normalization_4/AssignNewValue�&batch_normalization_4/AssignNewValue_1�$batch_normalization_5/AssignNewValue�&batch_normalization_5/AssignNewValue_1�$batch_normalization_6/AssignNewValue�&batch_normalization_6/AssignNewValue_1�$batch_normalization_7/AssignNewValue�&batch_normalization_7/AssignNewValue_1T
reshape/ShapeShapeinputs*
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
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
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
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2
reshape/Reshaped
covtr4/ShapeShapereshape/Reshape:output:0*
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
value	B :2
covtr4/stack/1b
covtr4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
covtr4/stack/2b
covtr4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
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
:*
dtype02(
&covtr4/conv2d_transpose/ReadVariableOp�
covtr4/conv2d_transposeConv2DBackpropInputcovtr4/stack:output:0.covtr4/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
covtr4/conv2d_transpose�
covtr4/BiasAdd/ReadVariableOpReadVariableOp&covtr4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr4/BiasAdd/ReadVariableOp�
covtr4/BiasAddBiasAdd covtr4/conv2d_transpose:output:0%covtr4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
covtr4/BiasAdd�
leaky_re_lu/LeakyRelu	LeakyRelucovtr4/BiasAdd:output:0*/
_output_shapes
:���������2
leaky_re_lu/LeakyRelu�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3#leaky_re_lu/LeakyRelu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2&
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
covtr5/ShapeShape(batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr5/Shape�
covtr5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr5/strided_slice/stack�
covtr5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr5/strided_slice/stack_1�
covtr5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr5/strided_slice/stack_2�
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
value	B :2
covtr5/stack/1b
covtr5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
covtr5/stack/2b
covtr5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr5/stack/3�
covtr5/stackPackcovtr5/strided_slice:output:0covtr5/stack/1:output:0covtr5/stack/2:output:0covtr5/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr5/stack�
covtr5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr5/strided_slice_1/stack�
covtr5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr5/strided_slice_1/stack_1�
covtr5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr5/strided_slice_1/stack_2�
covtr5/strided_slice_1StridedSlicecovtr5/stack:output:0%covtr5/strided_slice_1/stack:output:0'covtr5/strided_slice_1/stack_1:output:0'covtr5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr5/strided_slice_1�
&covtr5/conv2d_transpose/ReadVariableOpReadVariableOp/covtr5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr5/conv2d_transpose/ReadVariableOp�
covtr5/conv2d_transposeConv2DBackpropInputcovtr5/stack:output:0.covtr5/conv2d_transpose/ReadVariableOp:value:0(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
covtr5/conv2d_transpose�
covtr5/BiasAdd/ReadVariableOpReadVariableOp&covtr5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr5/BiasAdd/ReadVariableOp�
covtr5/BiasAddBiasAdd covtr5/conv2d_transpose:output:0%covtr5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
covtr5/BiasAdd�
leaky_re_lu_1/LeakyRelu	LeakyRelucovtr5/BiasAdd:output:0*/
_output_shapes
:���������2
leaky_re_lu_1/LeakyRelu�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_1/LeakyRelu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2(
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
covtr6/ShapeShape*batch_normalization_1/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr6/Shape�
covtr6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr6/strided_slice/stack�
covtr6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr6/strided_slice/stack_1�
covtr6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr6/strided_slice/stack_2�
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
value	B :2
covtr6/stack/1b
covtr6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
covtr6/stack/2b
covtr6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr6/stack/3�
covtr6/stackPackcovtr6/strided_slice:output:0covtr6/stack/1:output:0covtr6/stack/2:output:0covtr6/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr6/stack�
covtr6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr6/strided_slice_1/stack�
covtr6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr6/strided_slice_1/stack_1�
covtr6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr6/strided_slice_1/stack_2�
covtr6/strided_slice_1StridedSlicecovtr6/stack:output:0%covtr6/strided_slice_1/stack:output:0'covtr6/strided_slice_1/stack_1:output:0'covtr6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr6/strided_slice_1�
&covtr6/conv2d_transpose/ReadVariableOpReadVariableOp/covtr6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr6/conv2d_transpose/ReadVariableOp�
covtr6/conv2d_transposeConv2DBackpropInputcovtr6/stack:output:0.covtr6/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
covtr6/conv2d_transpose�
covtr6/BiasAdd/ReadVariableOpReadVariableOp&covtr6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr6/BiasAdd/ReadVariableOp�
covtr6/BiasAddBiasAdd covtr6/conv2d_transpose:output:0%covtr6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
covtr6/BiasAdd�
leaky_re_lu_2/LeakyRelu	LeakyRelucovtr6/BiasAdd:output:0*/
_output_shapes
:���������2
leaky_re_lu_2/LeakyRelu�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_2/LeakyRelu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2(
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
&batch_normalization_2/AssignNewValue_1v
covtr7/ShapeShape*batch_normalization_2/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr7/Shape�
covtr7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr7/strided_slice/stack�
covtr7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr7/strided_slice/stack_1�
covtr7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr7/strided_slice/stack_2�
covtr7/strided_sliceStridedSlicecovtr7/Shape:output:0#covtr7/strided_slice/stack:output:0%covtr7/strided_slice/stack_1:output:0%covtr7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr7/strided_sliceb
covtr7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
covtr7/stack/1b
covtr7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
covtr7/stack/2b
covtr7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr7/stack/3�
covtr7/stackPackcovtr7/strided_slice:output:0covtr7/stack/1:output:0covtr7/stack/2:output:0covtr7/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr7/stack�
covtr7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr7/strided_slice_1/stack�
covtr7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr7/strided_slice_1/stack_1�
covtr7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr7/strided_slice_1/stack_2�
covtr7/strided_slice_1StridedSlicecovtr7/stack:output:0%covtr7/strided_slice_1/stack:output:0'covtr7/strided_slice_1/stack_1:output:0'covtr7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr7/strided_slice_1�
&covtr7/conv2d_transpose/ReadVariableOpReadVariableOp/covtr7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr7/conv2d_transpose/ReadVariableOp�
covtr7/conv2d_transposeConv2DBackpropInputcovtr7/stack:output:0.covtr7/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
covtr7/conv2d_transpose�
covtr7/BiasAdd/ReadVariableOpReadVariableOp&covtr7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr7/BiasAdd/ReadVariableOp�
covtr7/BiasAddBiasAdd covtr7/conv2d_transpose:output:0%covtr7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
covtr7/BiasAdd�
leaky_re_lu_3/LeakyRelu	LeakyRelucovtr7/BiasAdd:output:0*/
_output_shapes
:���������2
leaky_re_lu_3/LeakyRelu�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_3/ReadVariableOp�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_3/ReadVariableOp_1�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_3/LeakyRelu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_3/FusedBatchNormV3�
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue�
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1v
covtr8/ShapeShape*batch_normalization_3/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr8/Shape�
covtr8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr8/strided_slice/stack�
covtr8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr8/strided_slice/stack_1�
covtr8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr8/strided_slice/stack_2�
covtr8/strided_sliceStridedSlicecovtr8/Shape:output:0#covtr8/strided_slice/stack:output:0%covtr8/strided_slice/stack_1:output:0%covtr8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr8/strided_sliceb
covtr8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
covtr8/stack/1b
covtr8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
covtr8/stack/2b
covtr8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr8/stack/3�
covtr8/stackPackcovtr8/strided_slice:output:0covtr8/stack/1:output:0covtr8/stack/2:output:0covtr8/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr8/stack�
covtr8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr8/strided_slice_1/stack�
covtr8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr8/strided_slice_1/stack_1�
covtr8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr8/strided_slice_1/stack_2�
covtr8/strided_slice_1StridedSlicecovtr8/stack:output:0%covtr8/strided_slice_1/stack:output:0'covtr8/strided_slice_1/stack_1:output:0'covtr8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr8/strided_slice_1�
&covtr8/conv2d_transpose/ReadVariableOpReadVariableOp/covtr8_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr8/conv2d_transpose/ReadVariableOp�
covtr8/conv2d_transposeConv2DBackpropInputcovtr8/stack:output:0.covtr8/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
covtr8/conv2d_transpose�
covtr8/BiasAdd/ReadVariableOpReadVariableOp&covtr8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr8/BiasAdd/ReadVariableOp�
covtr8/BiasAddBiasAdd covtr8/conv2d_transpose:output:0%covtr8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
covtr8/BiasAdd�
leaky_re_lu_4/LeakyRelu	LeakyRelucovtr8/BiasAdd:output:0*/
_output_shapes
:���������2
leaky_re_lu_4/LeakyRelu�
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_4/ReadVariableOp�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_4/LeakyRelu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_4/FusedBatchNormV3�
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue�
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1v
covtr9/ShapeShape*batch_normalization_4/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr9/Shape�
covtr9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr9/strided_slice/stack�
covtr9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr9/strided_slice/stack_1�
covtr9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr9/strided_slice/stack_2�
covtr9/strided_sliceStridedSlicecovtr9/Shape:output:0#covtr9/strided_slice/stack:output:0%covtr9/strided_slice/stack_1:output:0%covtr9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr9/strided_sliceb
covtr9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :$2
covtr9/stack/1b
covtr9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
covtr9/stack/2b
covtr9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr9/stack/3�
covtr9/stackPackcovtr9/strided_slice:output:0covtr9/stack/1:output:0covtr9/stack/2:output:0covtr9/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr9/stack�
covtr9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr9/strided_slice_1/stack�
covtr9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr9/strided_slice_1/stack_1�
covtr9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr9/strided_slice_1/stack_2�
covtr9/strided_slice_1StridedSlicecovtr9/stack:output:0%covtr9/strided_slice_1/stack:output:0'covtr9/strided_slice_1/stack_1:output:0'covtr9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr9/strided_slice_1�
&covtr9/conv2d_transpose/ReadVariableOpReadVariableOp/covtr9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr9/conv2d_transpose/ReadVariableOp�
covtr9/conv2d_transposeConv2DBackpropInputcovtr9/stack:output:0.covtr9/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������$*
paddingVALID*
strides
2
covtr9/conv2d_transpose�
covtr9/BiasAdd/ReadVariableOpReadVariableOp&covtr9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr9/BiasAdd/ReadVariableOp�
covtr9/BiasAddBiasAdd covtr9/conv2d_transpose:output:0%covtr9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������$2
covtr9/BiasAdd�
leaky_re_lu_5/LeakyRelu	LeakyRelucovtr9/BiasAdd:output:0*/
_output_shapes
:���������$2
leaky_re_lu_5/LeakyRelu�
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_5/ReadVariableOp�
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_5/ReadVariableOp_1�
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_5/LeakyRelu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������$:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_5/FusedBatchNormV3�
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue�
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1x
covtr10/ShapeShape*batch_normalization_5/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr10/Shape�
covtr10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr10/strided_slice/stack�
covtr10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr10/strided_slice/stack_1�
covtr10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr10/strided_slice/stack_2�
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
value	B :+2
covtr10/stack/1d
covtr10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :!2
covtr10/stack/2d
covtr10/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr10/stack/3�
covtr10/stackPackcovtr10/strided_slice:output:0covtr10/stack/1:output:0covtr10/stack/2:output:0covtr10/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr10/stack�
covtr10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr10/strided_slice_1/stack�
covtr10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr10/strided_slice_1/stack_1�
covtr10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr10/strided_slice_1/stack_2�
covtr10/strided_slice_1StridedSlicecovtr10/stack:output:0&covtr10/strided_slice_1/stack:output:0(covtr10/strided_slice_1/stack_1:output:0(covtr10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr10/strided_slice_1�
'covtr10/conv2d_transpose/ReadVariableOpReadVariableOp0covtr10_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02)
'covtr10/conv2d_transpose/ReadVariableOp�
covtr10/conv2d_transposeConv2DBackpropInputcovtr10/stack:output:0/covtr10/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������+!*
paddingVALID*
strides
2
covtr10/conv2d_transpose�
covtr10/BiasAdd/ReadVariableOpReadVariableOp'covtr10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
covtr10/BiasAdd/ReadVariableOp�
covtr10/BiasAddBiasAdd!covtr10/conv2d_transpose:output:0&covtr10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������+!2
covtr10/BiasAdd�
leaky_re_lu_6/LeakyRelu	LeakyRelucovtr10/BiasAdd:output:0*/
_output_shapes
:���������+!2
leaky_re_lu_6/LeakyRelu�
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_6/ReadVariableOp�
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1�
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_6/LeakyRelu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������+!:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_6/FusedBatchNormV3�
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValue�
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1x
covtr11/ShapeShape*batch_normalization_6/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr11/Shape�
covtr11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr11/strided_slice/stack�
covtr11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr11/strided_slice/stack_1�
covtr11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr11/strided_slice/stack_2�
covtr11/strided_sliceStridedSlicecovtr11/Shape:output:0$covtr11/strided_slice/stack:output:0&covtr11/strided_slice/stack_1:output:0&covtr11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr11/strided_sliced
covtr11/stack/1Const*
_output_shapes
: *
dtype0*
value	B :32
covtr11/stack/1d
covtr11/stack/2Const*
_output_shapes
: *
dtype0*
value	B :)2
covtr11/stack/2d
covtr11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :"2
covtr11/stack/3�
covtr11/stackPackcovtr11/strided_slice:output:0covtr11/stack/1:output:0covtr11/stack/2:output:0covtr11/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr11/stack�
covtr11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr11/strided_slice_1/stack�
covtr11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr11/strided_slice_1/stack_1�
covtr11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr11/strided_slice_1/stack_2�
covtr11/strided_slice_1StridedSlicecovtr11/stack:output:0&covtr11/strided_slice_1/stack:output:0(covtr11/strided_slice_1/stack_1:output:0(covtr11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr11/strided_slice_1�
'covtr11/conv2d_transpose/ReadVariableOpReadVariableOp0covtr11_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		"*
dtype02)
'covtr11/conv2d_transpose/ReadVariableOp�
covtr11/conv2d_transposeConv2DBackpropInputcovtr11/stack:output:0/covtr11/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������3)"*
paddingVALID*
strides
2
covtr11/conv2d_transpose�
covtr11/BiasAdd/ReadVariableOpReadVariableOp'covtr11_biasadd_readvariableop_resource*
_output_shapes
:"*
dtype02 
covtr11/BiasAdd/ReadVariableOp�
covtr11/BiasAddBiasAdd!covtr11/conv2d_transpose:output:0&covtr11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������3)"2
covtr11/BiasAdd�
leaky_re_lu_7/LeakyRelu	LeakyRelucovtr11/BiasAdd:output:0*/
_output_shapes
:���������3)"2
leaky_re_lu_7/LeakyRelu�
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:"*
dtype02&
$batch_normalization_7/ReadVariableOp�
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:"*
dtype02(
&batch_normalization_7/ReadVariableOp_1�
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:"*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:"*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_7/LeakyRelu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������3)":":":":":*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_7/FusedBatchNormV3�
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue�
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1x
covtr14/ShapeShape*batch_normalization_7/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr14/Shape�
covtr14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr14/strided_slice/stack�
covtr14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr14/strided_slice/stack_1�
covtr14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr14/strided_slice/stack_2�
covtr14/strided_sliceStridedSlicecovtr14/Shape:output:0$covtr14/strided_slice/stack:output:0&covtr14/strided_slice/stack_1:output:0&covtr14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr14/strided_sliced
covtr14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :72
covtr14/stack/1d
covtr14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :-2
covtr14/stack/2d
covtr14/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
covtr14/stack/3�
covtr14/stackPackcovtr14/strided_slice:output:0covtr14/stack/1:output:0covtr14/stack/2:output:0covtr14/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr14/stack�
covtr14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr14/strided_slice_1/stack�
covtr14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr14/strided_slice_1/stack_1�
covtr14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr14/strided_slice_1/stack_2�
covtr14/strided_slice_1StridedSlicecovtr14/stack:output:0&covtr14/strided_slice_1/stack:output:0(covtr14/strided_slice_1/stack_1:output:0(covtr14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr14/strided_slice_1�
'covtr14/conv2d_transpose/ReadVariableOpReadVariableOp0covtr14_conv2d_transpose_readvariableop_resource*&
_output_shapes
:"*
dtype02)
'covtr14/conv2d_transpose/ReadVariableOp�
covtr14/conv2d_transposeConv2DBackpropInputcovtr14/stack:output:0/covtr14/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������7-*
paddingVALID*
strides
2
covtr14/conv2d_transpose�
covtr14/BiasAdd/ReadVariableOpReadVariableOp'covtr14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
covtr14/BiasAdd/ReadVariableOp�
covtr14/BiasAddBiasAdd!covtr14/conv2d_transpose:output:0&covtr14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������7-2
covtr14/BiasAdd�
covtr14/SigmoidSigmoidcovtr14/BiasAdd:output:0*
T0*/
_output_shapes
:���������7-2
covtr14/Sigmoid�
IdentityIdentitycovtr14/Sigmoid:y:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_1*
T0*/
_output_shapes
:���������7-2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������::::::::::::::::::::::::::::::::::::::::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_1:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_96155

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
|
'__inference_covtr14_layer_call_fn_93790

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
GPU2 *0J 8� *K
fFRD
B__inference_covtr14_layer_call_and_return_conditional_losses_937802
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������"::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������"
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_96211

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
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
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_93932

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_94197

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������"2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������"2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������":i e
A
_output_shapes/
-:+���������������������������"
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_96359

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:"*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:"*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:"*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:"*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������":":":":":*
epsilon%o�:*
exponential_avg_factor%
�#<2
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
-:+���������������������������"2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������"::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������"
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_96186

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_96229

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�f
�
__inference__traced_save_96576
file_prefix,
(savev2_covtr4_kernel_read_readvariableop*
&savev2_covtr4_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop,
(savev2_covtr5_kernel_read_readvariableop*
&savev2_covtr5_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop,
(savev2_covtr6_kernel_read_readvariableop*
&savev2_covtr6_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop,
(savev2_covtr7_kernel_read_readvariableop*
&savev2_covtr7_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop,
(savev2_covtr8_kernel_read_readvariableop*
&savev2_covtr8_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop,
(savev2_covtr9_kernel_read_readvariableop*
&savev2_covtr9_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop-
)savev2_covtr10_kernel_read_readvariableop+
'savev2_covtr10_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop-
)savev2_covtr11_kernel_read_readvariableop+
'savev2_covtr11_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop-
)savev2_covtr14_kernel_read_readvariableop+
'savev2_covtr14_bias_read_readvariableop
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
value3B1 B+_temp_67feee03e4304ea2aca3d9ea0d51ea99/part2	
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
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*�
value�B�3B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_covtr4_kernel_read_readvariableop&savev2_covtr4_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop(savev2_covtr5_kernel_read_readvariableop&savev2_covtr5_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop(savev2_covtr6_kernel_read_readvariableop&savev2_covtr6_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop(savev2_covtr7_kernel_read_readvariableop&savev2_covtr7_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop(savev2_covtr8_kernel_read_readvariableop&savev2_covtr8_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop(savev2_covtr9_kernel_read_readvariableop&savev2_covtr9_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop)savev2_covtr10_kernel_read_readvariableop'savev2_covtr10_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop)savev2_covtr11_kernel_read_readvariableop'savev2_covtr11_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop)savev2_covtr14_kernel_read_readvariableop'savev2_covtr14_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *A
dtypes7
5232
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::		":":":":":":":: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
::,%(
&
_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
::,+(
&
_output_shapes
:		": ,

_output_shapes
:": -

_output_shapes
:": .

_output_shapes
:": /

_output_shapes
:": 0

_output_shapes
:":,1(
&
_output_shapes
:": 2

_output_shapes
::3

_output_shapes
: 
�
I
-__inference_leaky_re_lu_1_layer_call_fn_95895

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
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_938792
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�$
�
A__inference_covtr6_layer_call_and_return_conditional_losses_92867

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
value	B :2	
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
value	B :2	
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
:*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
@
	gen_noise3
serving_default_gen_noise:0����������C
covtr148
StatefulPartitionedCall:0���������7-tensorflow/serving/predict:��
��
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer-15
layer_with_weights-9
layer-16
layer_with_weights-10
layer-17
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer-21
layer_with_weights-13
layer-22
layer_with_weights-14
layer-23
layer-24
layer_with_weights-15
layer-25
layer_with_weights-16
layer-26
	variables
regularization_losses
trainable_variables
	keras_api
 
signatures
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"��
_tf_keras_network��{"class_name": "Functional", "name": "Generator", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Generator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 225]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gen_noise"}, "name": "gen_noise", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [15, 5, 3]}}, "name": "reshape", "inbound_nodes": [[["gen_noise", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr4", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr4", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu", "inbound_nodes": [[["covtr4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr5", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr5", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_1", "inbound_nodes": [[["covtr5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr6", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr6", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_2", "inbound_nodes": [[["covtr6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr7", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr7", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_3", "inbound_nodes": [[["covtr7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["leaky_re_lu_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr8", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr8", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_4", "inbound_nodes": [[["covtr8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr9", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr9", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_5", "inbound_nodes": [[["covtr9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["leaky_re_lu_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr10", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr10", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_6", "inbound_nodes": [[["covtr10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["leaky_re_lu_6", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr11", "trainable": true, "dtype": "float32", "filters": 34, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr11", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_7", "inbound_nodes": [[["covtr11", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["leaky_re_lu_7", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr14", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr14", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}], "input_layers": [["gen_noise", 0, 0]], "output_layers": [["covtr14", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 225]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Generator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 225]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gen_noise"}, "name": "gen_noise", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [15, 5, 3]}}, "name": "reshape", "inbound_nodes": [[["gen_noise", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr4", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr4", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu", "inbound_nodes": [[["covtr4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr5", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr5", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_1", "inbound_nodes": [[["covtr5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr6", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr6", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_2", "inbound_nodes": [[["covtr6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr7", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr7", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_3", "inbound_nodes": [[["covtr7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["leaky_re_lu_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr8", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr8", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_4", "inbound_nodes": [[["covtr8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr9", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr9", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_5", "inbound_nodes": [[["covtr9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["leaky_re_lu_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr10", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr10", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_6", "inbound_nodes": [[["covtr10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["leaky_re_lu_6", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr11", "trainable": true, "dtype": "float32", "filters": 34, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr11", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_7", "inbound_nodes": [[["covtr11", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["leaky_re_lu_7", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr14", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr14", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}], "input_layers": [["gen_noise", 0, 0]], "output_layers": [["covtr14", 0, 0]]}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "gen_noise", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 225]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 225]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gen_noise"}}
�
!	variables
"regularization_losses
#trainable_variables
$	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [15, 5, 3]}}}
�


%kernel
&bias
'	variables
(regularization_losses
)trainable_variables
*	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "covtr4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr4", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 5, 3]}}
�
+	variables
,regularization_losses
-trainable_variables
.	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
�	
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4	variables
5regularization_losses
6trainable_variables
7	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 6, 4]}}
�


8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "covtr5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr5", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 6, 4]}}
�
>	variables
?regularization_losses
@trainable_variables
A	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
�	
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18, 8, 6]}}
�


Kkernel
Lbias
M	variables
Nregularization_losses
Otrainable_variables
P	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "covtr6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr6", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18, 8, 6]}}
�
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
�	
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21, 11, 8]}}
�


^kernel
_bias
`	variables
aregularization_losses
btrainable_variables
c	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "covtr7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr7", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21, 11, 8]}}
�
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
�	
haxis
	igamma
jbeta
kmoving_mean
lmoving_variance
m	variables
nregularization_losses
otrainable_variables
p	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25, 15, 16]}}
�


qkernel
rbias
s	variables
tregularization_losses
utrainable_variables
v	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "covtr8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr8", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25, 15, 16]}}
�
w	variables
xregularization_losses
ytrainable_variables
z	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
�	
{axis
	|gamma
}beta
~moving_mean
moving_variance
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 20, 20]}}
�

�kernel
	�bias
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "covtr9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr9", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 20, 20]}}
�
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
�	
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36, 26, 24]}}
�

�kernel
	�bias
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "covtr10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr10", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36, 26, 24]}}
�
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
�	
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 43, 33, 30]}}
�

�kernel
	�bias
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "covtr11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr11", "trainable": true, "dtype": "float32", "filters": 34, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 43, 33, 30]}}
�
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
�	
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 34}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 51, 41, 34]}}
�

�kernel
	�bias
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "covtr14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr14", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 34}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 51, 41, 34]}}
�
%0
&1
02
13
24
35
86
97
C8
D9
E10
F11
K12
L13
V14
W15
X16
Y17
^18
_19
i20
j21
k22
l23
q24
r25
|26
}27
~28
29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49"
trackable_list_wrapper
 "
trackable_list_wrapper
�
%0
&1
02
13
84
95
C6
D7
K8
L9
V10
W11
^12
_13
i14
j15
q16
r17
|18
}19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33"
trackable_list_wrapper
�
	variables
�non_trainable_variables
regularization_losses
 �layer_regularization_losses
�layers
trainable_variables
�layer_metrics
�metrics
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
!	variables
�non_trainable_variables
"regularization_losses
 �layer_regularization_losses
�layers
#trainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':%2covtr4/kernel
:2covtr4/bias
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
�
'	variables
�non_trainable_variables
(regularization_losses
 �layer_regularization_losses
�layers
)trainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
+	variables
�non_trainable_variables
,regularization_losses
 �layer_regularization_losses
�layers
-trainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
<
00
11
22
33"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
�
4	variables
�non_trainable_variables
5regularization_losses
 �layer_regularization_losses
�layers
6trainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':%2covtr5/kernel
:2covtr5/bias
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
�
:	variables
�non_trainable_variables
;regularization_losses
 �layer_regularization_losses
�layers
<trainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
>	variables
�non_trainable_variables
?regularization_losses
 �layer_regularization_losses
�layers
@trainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_1/gamma
(:&2batch_normalization_1/beta
1:/ (2!batch_normalization_1/moving_mean
5:3 (2%batch_normalization_1/moving_variance
<
C0
D1
E2
F3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
�
G	variables
�non_trainable_variables
Hregularization_losses
 �layer_regularization_losses
�layers
Itrainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':%2covtr6/kernel
:2covtr6/bias
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
�
M	variables
�non_trainable_variables
Nregularization_losses
 �layer_regularization_losses
�layers
Otrainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Q	variables
�non_trainable_variables
Rregularization_losses
 �layer_regularization_losses
�layers
Strainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_2/gamma
(:&2batch_normalization_2/beta
1:/ (2!batch_normalization_2/moving_mean
5:3 (2%batch_normalization_2/moving_variance
<
V0
W1
X2
Y3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
�
Z	variables
�non_trainable_variables
[regularization_losses
 �layer_regularization_losses
�layers
\trainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':%2covtr7/kernel
:2covtr7/bias
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
�
`	variables
�non_trainable_variables
aregularization_losses
 �layer_regularization_losses
�layers
btrainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
d	variables
�non_trainable_variables
eregularization_losses
 �layer_regularization_losses
�layers
ftrainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_3/gamma
(:&2batch_normalization_3/beta
1:/ (2!batch_normalization_3/moving_mean
5:3 (2%batch_normalization_3/moving_variance
<
i0
j1
k2
l3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
�
m	variables
�non_trainable_variables
nregularization_losses
 �layer_regularization_losses
�layers
otrainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':%2covtr8/kernel
:2covtr8/bias
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
�
s	variables
�non_trainable_variables
tregularization_losses
 �layer_regularization_losses
�layers
utrainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
w	variables
�non_trainable_variables
xregularization_losses
 �layer_regularization_losses
�layers
ytrainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_4/gamma
(:&2batch_normalization_4/beta
1:/ (2!batch_normalization_4/moving_mean
5:3 (2%batch_normalization_4/moving_variance
<
|0
}1
~2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':%2covtr9/kernel
:2covtr9/bias
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_5/gamma
(:&2batch_normalization_5/beta
1:/ (2!batch_normalization_5/moving_mean
5:3 (2%batch_normalization_5/moving_variance
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(:&2covtr10/kernel
:2covtr10/bias
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_6/gamma
(:&2batch_normalization_6/beta
1:/ (2!batch_normalization_6/moving_mean
5:3 (2%batch_normalization_6/moving_variance
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(:&		"2covtr11/kernel
:"2covtr11/bias
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'"2batch_normalization_7/gamma
(:&"2batch_normalization_7/beta
1:/" (2!batch_normalization_7/moving_mean
5:3" (2%batch_normalization_7/moving_variance
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(:&"2covtr14/kernel
:2covtr14/bias
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�	variables
�non_trainable_variables
�regularization_losses
 �layer_regularization_losses
�layers
�trainable_variables
�layer_metrics
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
20
31
E2
F3
X4
Y5
k6
l7
~8
9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
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
16
17
18
19
20
21
22
23
24
25
26"
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
20
31"
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
E0
F1"
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
X0
Y1"
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
k0
l1"
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
~0
1"
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
0
�0
�1"
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
0
�0
�1"
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
0
�0
�1"
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
�2�
D__inference_Generator_layer_call_and_return_conditional_losses_95582
D__inference_Generator_layer_call_and_return_conditional_losses_94246
D__inference_Generator_layer_call_and_return_conditional_losses_94376
D__inference_Generator_layer_call_and_return_conditional_losses_95276�
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
 __inference__wrapped_model_92525�
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
annotations� *)�&
$�!
	gen_noise����������
�2�
)__inference_Generator_layer_call_fn_94612
)__inference_Generator_layer_call_fn_95792
)__inference_Generator_layer_call_fn_94847
)__inference_Generator_layer_call_fn_95687�
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
B__inference_reshape_layer_call_and_return_conditional_losses_95806�
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
'__inference_reshape_layer_call_fn_95811�
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
A__inference_covtr4_layer_call_and_return_conditional_losses_92563�
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
&__inference_covtr4_layer_call_fn_92573�
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
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_95816�
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
+__inference_leaky_re_lu_layer_call_fn_95821�
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
N__inference_batch_normalization_layer_call_and_return_conditional_losses_95859
N__inference_batch_normalization_layer_call_and_return_conditional_losses_95841�
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
3__inference_batch_normalization_layer_call_fn_95885
3__inference_batch_normalization_layer_call_fn_95872�
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
A__inference_covtr5_layer_call_and_return_conditional_losses_92715�
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
2�/+���������������������������
�2�
&__inference_covtr5_layer_call_fn_92725�
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
2�/+���������������������������
�2�
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_95890�
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
-__inference_leaky_re_lu_1_layer_call_fn_95895�
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
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_95915
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_95933�
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
5__inference_batch_normalization_1_layer_call_fn_95946
5__inference_batch_normalization_1_layer_call_fn_95959�
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
A__inference_covtr6_layer_call_and_return_conditional_losses_92867�
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
2�/+���������������������������
�2�
&__inference_covtr6_layer_call_fn_92877�
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
2�/+���������������������������
�2�
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_95964�
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
-__inference_leaky_re_lu_2_layer_call_fn_95969�
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
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_95989
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_96007�
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
5__inference_batch_normalization_2_layer_call_fn_96020
5__inference_batch_normalization_2_layer_call_fn_96033�
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
A__inference_covtr7_layer_call_and_return_conditional_losses_93019�
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
2�/+���������������������������
�2�
&__inference_covtr7_layer_call_fn_93029�
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
2�/+���������������������������
�2�
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_96038�
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
-__inference_leaky_re_lu_3_layer_call_fn_96043�
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
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_96081
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_96063�
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
5__inference_batch_normalization_3_layer_call_fn_96094
5__inference_batch_normalization_3_layer_call_fn_96107�
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
A__inference_covtr8_layer_call_and_return_conditional_losses_93171�
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
2�/+���������������������������
�2�
&__inference_covtr8_layer_call_fn_93181�
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
2�/+���������������������������
�2�
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_96112�
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
-__inference_leaky_re_lu_4_layer_call_fn_96117�
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
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_96137
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_96155�
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
5__inference_batch_normalization_4_layer_call_fn_96181
5__inference_batch_normalization_4_layer_call_fn_96168�
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
A__inference_covtr9_layer_call_and_return_conditional_losses_93323�
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
2�/+���������������������������
�2�
&__inference_covtr9_layer_call_fn_93333�
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
2�/+���������������������������
�2�
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_96186�
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
-__inference_leaky_re_lu_5_layer_call_fn_96191�
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
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_96211
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_96229�
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
5__inference_batch_normalization_5_layer_call_fn_96242
5__inference_batch_normalization_5_layer_call_fn_96255�
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
B__inference_covtr10_layer_call_and_return_conditional_losses_93475�
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
2�/+���������������������������
�2�
'__inference_covtr10_layer_call_fn_93485�
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
2�/+���������������������������
�2�
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_96260�
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
-__inference_leaky_re_lu_6_layer_call_fn_96265�
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
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_96285
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_96303�
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
5__inference_batch_normalization_6_layer_call_fn_96316
5__inference_batch_normalization_6_layer_call_fn_96329�
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
B__inference_covtr11_layer_call_and_return_conditional_losses_93627�
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
2�/+���������������������������
�2�
'__inference_covtr11_layer_call_fn_93637�
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
2�/+���������������������������
�2�
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_96334�
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
-__inference_leaky_re_lu_7_layer_call_fn_96339�
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
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_96359
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_96377�
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
5__inference_batch_normalization_7_layer_call_fn_96390
5__inference_batch_normalization_7_layer_call_fn_96403�
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
B__inference_covtr14_layer_call_and_return_conditional_losses_93780�
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
2�/+���������������������������"
�2�
'__inference_covtr14_layer_call_fn_93790�
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
2�/+���������������������������"
4B2
#__inference_signature_wrapper_94954	gen_noise�
D__inference_Generator_layer_call_and_return_conditional_losses_94246�F%&012389CDEFKLVWXY^_ijklqr|}~��������������������;�8
1�.
$�!
	gen_noise����������
p

 
� "?�<
5�2
0+���������������������������
� �
D__inference_Generator_layer_call_and_return_conditional_losses_94376�F%&012389CDEFKLVWXY^_ijklqr|}~��������������������;�8
1�.
$�!
	gen_noise����������
p 

 
� "?�<
5�2
0+���������������������������
� �
D__inference_Generator_layer_call_and_return_conditional_losses_95276�F%&012389CDEFKLVWXY^_ijklqr|}~��������������������8�5
.�+
!�
inputs����������
p

 
� "-�*
#� 
0���������7-
� �
D__inference_Generator_layer_call_and_return_conditional_losses_95582�F%&012389CDEFKLVWXY^_ijklqr|}~��������������������8�5
.�+
!�
inputs����������
p 

 
� "-�*
#� 
0���������7-
� �
)__inference_Generator_layer_call_fn_94612�F%&012389CDEFKLVWXY^_ijklqr|}~��������������������;�8
1�.
$�!
	gen_noise����������
p

 
� "2�/+����������������������������
)__inference_Generator_layer_call_fn_94847�F%&012389CDEFKLVWXY^_ijklqr|}~��������������������;�8
1�.
$�!
	gen_noise����������
p 

 
� "2�/+����������������������������
)__inference_Generator_layer_call_fn_95687�F%&012389CDEFKLVWXY^_ijklqr|}~��������������������8�5
.�+
!�
inputs����������
p

 
� "2�/+����������������������������
)__inference_Generator_layer_call_fn_95792�F%&012389CDEFKLVWXY^_ijklqr|}~��������������������8�5
.�+
!�
inputs����������
p 

 
� "2�/+����������������������������
 __inference__wrapped_model_92525�F%&012389CDEFKLVWXY^_ijklqr|}~��������������������3�0
)�&
$�!
	gen_noise����������
� "9�6
4
covtr14)�&
covtr14���������7-�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_95915�CDEFM�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_95933�CDEFM�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
5__inference_batch_normalization_1_layer_call_fn_95946�CDEFM�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
5__inference_batch_normalization_1_layer_call_fn_95959�CDEFM�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_95989�VWXYM�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_96007�VWXYM�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
5__inference_batch_normalization_2_layer_call_fn_96020�VWXYM�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
5__inference_batch_normalization_2_layer_call_fn_96033�VWXYM�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_96063�ijklM�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_96081�ijklM�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
5__inference_batch_normalization_3_layer_call_fn_96094�ijklM�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
5__inference_batch_normalization_3_layer_call_fn_96107�ijklM�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_96137�|}~M�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_96155�|}~M�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
5__inference_batch_normalization_4_layer_call_fn_96168�|}~M�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
5__inference_batch_normalization_4_layer_call_fn_96181�|}~M�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_96211�����M�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_96229�����M�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
5__inference_batch_normalization_5_layer_call_fn_96242�����M�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
5__inference_batch_normalization_5_layer_call_fn_96255�����M�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_96285�����M�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_96303�����M�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
5__inference_batch_normalization_6_layer_call_fn_96316�����M�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
5__inference_batch_normalization_6_layer_call_fn_96329�����M�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_96359�����M�J
C�@
:�7
inputs+���������������������������"
p
� "?�<
5�2
0+���������������������������"
� �
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_96377�����M�J
C�@
:�7
inputs+���������������������������"
p 
� "?�<
5�2
0+���������������������������"
� �
5__inference_batch_normalization_7_layer_call_fn_96390�����M�J
C�@
:�7
inputs+���������������������������"
p
� "2�/+���������������������������"�
5__inference_batch_normalization_7_layer_call_fn_96403�����M�J
C�@
:�7
inputs+���������������������������"
p 
� "2�/+���������������������������"�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_95841�0123M�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
N__inference_batch_normalization_layer_call_and_return_conditional_losses_95859�0123M�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
3__inference_batch_normalization_layer_call_fn_95872�0123M�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
3__inference_batch_normalization_layer_call_fn_95885�0123M�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
B__inference_covtr10_layer_call_and_return_conditional_losses_93475���I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
'__inference_covtr10_layer_call_fn_93485���I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
B__inference_covtr11_layer_call_and_return_conditional_losses_93627���I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������"
� �
'__inference_covtr11_layer_call_fn_93637���I�F
?�<
:�7
inputs+���������������������������
� "2�/+���������������������������"�
B__inference_covtr14_layer_call_and_return_conditional_losses_93780���I�F
?�<
:�7
inputs+���������������������������"
� "?�<
5�2
0+���������������������������
� �
'__inference_covtr14_layer_call_fn_93790���I�F
?�<
:�7
inputs+���������������������������"
� "2�/+����������������������������
A__inference_covtr4_layer_call_and_return_conditional_losses_92563�%&I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
&__inference_covtr4_layer_call_fn_92573�%&I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
A__inference_covtr5_layer_call_and_return_conditional_losses_92715�89I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
&__inference_covtr5_layer_call_fn_92725�89I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
A__inference_covtr6_layer_call_and_return_conditional_losses_92867�KLI�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
&__inference_covtr6_layer_call_fn_92877�KLI�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
A__inference_covtr7_layer_call_and_return_conditional_losses_93019�^_I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
&__inference_covtr7_layer_call_fn_93029�^_I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
A__inference_covtr8_layer_call_and_return_conditional_losses_93171�qrI�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
&__inference_covtr8_layer_call_fn_93181�qrI�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
A__inference_covtr9_layer_call_and_return_conditional_losses_93323���I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
&__inference_covtr9_layer_call_fn_93333���I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_95890�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
-__inference_leaky_re_lu_1_layer_call_fn_95895I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_95964�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
-__inference_leaky_re_lu_2_layer_call_fn_95969I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_96038�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
-__inference_leaky_re_lu_3_layer_call_fn_96043I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_96112�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
-__inference_leaky_re_lu_4_layer_call_fn_96117I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_96186�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
-__inference_leaky_re_lu_5_layer_call_fn_96191I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_96260�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
-__inference_leaky_re_lu_6_layer_call_fn_96265I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_96334�I�F
?�<
:�7
inputs+���������������������������"
� "?�<
5�2
0+���������������������������"
� �
-__inference_leaky_re_lu_7_layer_call_fn_96339I�F
?�<
:�7
inputs+���������������������������"
� "2�/+���������������������������"�
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_95816�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
+__inference_leaky_re_lu_layer_call_fn_95821I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
B__inference_reshape_layer_call_and_return_conditional_losses_95806a0�-
&�#
!�
inputs����������
� "-�*
#� 
0���������
� 
'__inference_reshape_layer_call_fn_95811T0�-
&�#
!�
inputs����������
� " �����������
#__inference_signature_wrapper_94954�F%&012389CDEFKLVWXY^_ijklqr|}~��������������������@�=
� 
6�3
1
	gen_noise$�!
	gen_noise����������"9�6
4
covtr14)�&
covtr14���������7-