└и"
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
 И"serve*2.3.12v2.3.0-54-gfcc4b966f18│╔
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
К
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
Г
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
И
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
Б
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
Ц
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
П
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
Ю
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
Ч
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
О
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma
З
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta
Е
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean
У
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance
Ы
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
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_2/moving_variance
Ы
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
О
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_3/gamma
З
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_3/beta
Е
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_3/moving_mean
У
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_3/moving_variance
Ы
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
О
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_4/gamma
З
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_4/beta
Е
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_4/moving_mean
У
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_4/moving_variance
Ы
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
О
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_5/gamma
З
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_5/beta
Е
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_5/moving_mean
У
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_5/moving_variance
Ы
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:*
dtype0
А
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
О
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_6/gamma
З
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_6/beta
Е
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_6/moving_mean
У
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_6/moving_variance
Ы
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:*
dtype0
А
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
О
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:"*,
shared_namebatch_normalization_7/gamma
З
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:"*
dtype0
М
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:"*+
shared_namebatch_normalization_7/beta
Е
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:"*
dtype0
Ъ
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:"*2
shared_name#!batch_normalization_7/moving_mean
У
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:"*
dtype0
в
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:"*6
shared_name'%batch_normalization_7/moving_variance
Ы
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:"*
dtype0
А
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
╧
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*К
valueАB¤~ BЎ~
У
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
trainable_variables
regularization_losses
	keras_api
 
signatures
 
R
!	variables
"trainable_variables
#regularization_losses
$	keras_api
h

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
R
+	variables
,trainable_variables
-regularization_losses
.	keras_api
Ч
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4	variables
5trainable_variables
6regularization_losses
7	keras_api
h

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
R
>	variables
?trainable_variables
@regularization_losses
A	keras_api
Ч
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
h

Kkernel
Lbias
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
R
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
Ч
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
h

^kernel
_bias
`	variables
atrainable_variables
bregularization_losses
c	keras_api
R
d	variables
etrainable_variables
fregularization_losses
g	keras_api
Ч
haxis
	igamma
jbeta
kmoving_mean
lmoving_variance
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
h

qkernel
rbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
R
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
Ы
{axis
	|gamma
}beta
~moving_mean
moving_variance
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
n
Дkernel
	Еbias
Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api
V
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
а
	Оaxis

Пgamma
	Рbeta
Сmoving_mean
Тmoving_variance
У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
n
Чkernel
	Шbias
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
V
Э	variables
Юtrainable_variables
Яregularization_losses
а	keras_api
а
	бaxis

вgamma
	гbeta
дmoving_mean
еmoving_variance
ж	variables
зtrainable_variables
иregularization_losses
й	keras_api
n
кkernel
	лbias
м	variables
нtrainable_variables
оregularization_losses
п	keras_api
V
░	variables
▒trainable_variables
▓regularization_losses
│	keras_api
а
	┤axis

╡gamma
	╢beta
╖moving_mean
╕moving_variance
╣	variables
║trainable_variables
╗regularization_losses
╝	keras_api
n
╜kernel
	╛bias
┐	variables
└trainable_variables
┴regularization_losses
┬	keras_api
Ъ
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
Д30
Е31
П32
Р33
С34
Т35
Ч36
Ш37
в38
г39
д40
е41
к42
л43
╡44
╢45
╖46
╕47
╜48
╛49
Ф
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
Д20
Е21
П22
Р23
Ч24
Ш25
в26
г27
к28
л29
╡30
╢31
╜32
╛33
 
▓
├layer_metrics
─non_trainable_variables
	variables
 ┼layer_regularization_losses
╞layers
trainable_variables
╟metrics
regularization_losses
 
 
 
 
▓
╚layer_metrics
╔non_trainable_variables
 ╩layer_regularization_losses
!	variables
╦layers
"trainable_variables
╠metrics
#regularization_losses
YW
VARIABLE_VALUEcovtr4/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr4/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1

%0
&1
 
▓
═layer_metrics
╬non_trainable_variables
 ╧layer_regularization_losses
'	variables
╨layers
(trainable_variables
╤metrics
)regularization_losses
 
 
 
▓
╥layer_metrics
╙non_trainable_variables
 ╘layer_regularization_losses
+	variables
╒layers
,trainable_variables
╓metrics
-regularization_losses
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

00
11
 
▓
╫layer_metrics
╪non_trainable_variables
 ┘layer_regularization_losses
4	variables
┌layers
5trainable_variables
█metrics
6regularization_losses
YW
VARIABLE_VALUEcovtr5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91

80
91
 
▓
▄layer_metrics
▌non_trainable_variables
 ▐layer_regularization_losses
:	variables
▀layers
;trainable_variables
рmetrics
<regularization_losses
 
 
 
▓
сlayer_metrics
тnon_trainable_variables
 уlayer_regularization_losses
>	variables
фlayers
?trainable_variables
хmetrics
@regularization_losses
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

C0
D1
 
▓
цlayer_metrics
чnon_trainable_variables
 шlayer_regularization_losses
G	variables
щlayers
Htrainable_variables
ъmetrics
Iregularization_losses
YW
VARIABLE_VALUEcovtr6/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr6/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

K0
L1

K0
L1
 
▓
ыlayer_metrics
ьnon_trainable_variables
 эlayer_regularization_losses
M	variables
юlayers
Ntrainable_variables
яmetrics
Oregularization_losses
 
 
 
▓
Ёlayer_metrics
ёnon_trainable_variables
 Єlayer_regularization_losses
Q	variables
єlayers
Rtrainable_variables
Їmetrics
Sregularization_losses
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

V0
W1
 
▓
їlayer_metrics
Ўnon_trainable_variables
 ўlayer_regularization_losses
Z	variables
°layers
[trainable_variables
∙metrics
\regularization_losses
YW
VARIABLE_VALUEcovtr7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr7/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

^0
_1

^0
_1
 
▓
·layer_metrics
√non_trainable_variables
 №layer_regularization_losses
`	variables
¤layers
atrainable_variables
■metrics
bregularization_losses
 
 
 
▓
 layer_metrics
Аnon_trainable_variables
 Бlayer_regularization_losses
d	variables
Вlayers
etrainable_variables
Гmetrics
fregularization_losses
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

i0
j1
 
▓
Дlayer_metrics
Еnon_trainable_variables
 Жlayer_regularization_losses
m	variables
Зlayers
ntrainable_variables
Иmetrics
oregularization_losses
YW
VARIABLE_VALUEcovtr8/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcovtr8/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

q0
r1

q0
r1
 
▓
Йlayer_metrics
Кnon_trainable_variables
 Лlayer_regularization_losses
s	variables
Мlayers
ttrainable_variables
Нmetrics
uregularization_losses
 
 
 
▓
Оlayer_metrics
Пnon_trainable_variables
 Рlayer_regularization_losses
w	variables
Сlayers
xtrainable_variables
Тmetrics
yregularization_losses
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

|0
}1
 
╡
Уlayer_metrics
Фnon_trainable_variables
 Хlayer_regularization_losses
А	variables
Цlayers
Бtrainable_variables
Чmetrics
Вregularization_losses
ZX
VARIABLE_VALUEcovtr9/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEcovtr9/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

Д0
Е1

Д0
Е1
 
╡
Шlayer_metrics
Щnon_trainable_variables
 Ъlayer_regularization_losses
Ж	variables
Ыlayers
Зtrainable_variables
Ьmetrics
Иregularization_losses
 
 
 
╡
Эlayer_metrics
Юnon_trainable_variables
 Яlayer_regularization_losses
К	variables
аlayers
Лtrainable_variables
бmetrics
Мregularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
П0
Р1
С2
Т3

П0
Р1
 
╡
вlayer_metrics
гnon_trainable_variables
 дlayer_regularization_losses
У	variables
еlayers
Фtrainable_variables
жmetrics
Хregularization_losses
[Y
VARIABLE_VALUEcovtr10/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEcovtr10/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

Ч0
Ш1

Ч0
Ш1
 
╡
зlayer_metrics
иnon_trainable_variables
 йlayer_regularization_losses
Щ	variables
кlayers
Ъtrainable_variables
лmetrics
Ыregularization_losses
 
 
 
╡
мlayer_metrics
нnon_trainable_variables
 оlayer_regularization_losses
Э	variables
пlayers
Юtrainable_variables
░metrics
Яregularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_6/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_6/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_6/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_6/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
в0
г1
д2
е3

в0
г1
 
╡
▒layer_metrics
▓non_trainable_variables
 │layer_regularization_losses
ж	variables
┤layers
зtrainable_variables
╡metrics
иregularization_losses
[Y
VARIABLE_VALUEcovtr11/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEcovtr11/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

к0
л1

к0
л1
 
╡
╢layer_metrics
╖non_trainable_variables
 ╕layer_regularization_losses
м	variables
╣layers
нtrainable_variables
║metrics
оregularization_losses
 
 
 
╡
╗layer_metrics
╝non_trainable_variables
 ╜layer_regularization_losses
░	variables
╛layers
▒trainable_variables
┐metrics
▓regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_7/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_7/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_7/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_7/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
╡0
╢1
╖2
╕3

╡0
╢1
 
╡
└layer_metrics
┴non_trainable_variables
 ┬layer_regularization_losses
╣	variables
├layers
║trainable_variables
─metrics
╗regularization_losses
[Y
VARIABLE_VALUEcovtr14/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEcovtr14/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

╜0
╛1

╜0
╛1
 
╡
┼layer_metrics
╞non_trainable_variables
 ╟layer_regularization_losses
┐	variables
╚layers
└trainable_variables
╔metrics
┴regularization_losses
 
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
С10
Т11
д12
е13
╖14
╕15
 
╬
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
С0
Т1
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
д0
е1
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
╖0
╕1
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
:         с*
dtype0*
shape:         с
╕
StatefulPartitionedCallStatefulPartitionedCallserving_default_gen_noisecovtr4/kernelcovtr4/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancecovtr5/kernelcovtr5/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancecovtr6/kernelcovtr6/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancecovtr7/kernelcovtr7/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancecovtr8/kernelcovtr8/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancecovtr9/kernelcovtr9/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variancecovtr10/kernelcovtr10/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancecovtr11/kernelcovtr11/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancecovtr14/kernelcovtr14/bias*>
Tin7
523*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         7-*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*2
config_proto" 

CPU

GPU2 *0J 8В *.
f)R'
%__inference_signature_wrapper_3415286
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
═
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
GPU2 *0J 8В *)
f$R"
 __inference__traced_save_3416908
р
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
GPU2 *0J 8В *,
f'R%
#__inference__traced_restore_3417068ьн
М
K
/__inference_leaky_re_lu_7_layer_call_fn_3416671

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
-:+                           "* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_34145292
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           "2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           ":i e
A
_output_shapes/
-:+                           "
 
_user_specified_nameinputs
╚
н
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3412967

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ъ
Л
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3416413

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ъ
Л
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_3413758

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╟
}
(__inference_covtr6_layer_call_fn_3413209

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
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr6_layer_call_and_return_conditional_losses_34131992
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
м
f
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_3416296

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ъ
Л
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3413454

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ў
·
+__inference_Generator_layer_call_fn_3415179
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
identityИвStatefulPartitionedCallп
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
-:+                           *T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_Generator_layer_call_and_return_conditional_losses_34150762
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         с::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
(
_output_shapes
:         с
#
_user_specified_name	gen_noise
╩
п
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3416691

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:"*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:"*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:"*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:"*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           ":":":":":*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
-:+                           "2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           "::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           "
 
_user_specified_nameinputs
д
к
7__inference_batch_normalization_4_layer_call_fn_3416500

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
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34135752
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
БК
╪
F__inference_Generator_layer_call_and_return_conditional_losses_3414578
	gen_noise
covtr4_3414148
covtr4_3414150
batch_normalization_3414192
batch_normalization_3414194
batch_normalization_3414196
batch_normalization_3414198
covtr5_3414201
covtr5_3414203!
batch_normalization_1_3414245!
batch_normalization_1_3414247!
batch_normalization_1_3414249!
batch_normalization_1_3414251
covtr6_3414254
covtr6_3414256!
batch_normalization_2_3414298!
batch_normalization_2_3414300!
batch_normalization_2_3414302!
batch_normalization_2_3414304
covtr7_3414307
covtr7_3414309!
batch_normalization_3_3414351!
batch_normalization_3_3414353!
batch_normalization_3_3414355!
batch_normalization_3_3414357
covtr8_3414360
covtr8_3414362!
batch_normalization_4_3414404!
batch_normalization_4_3414406!
batch_normalization_4_3414408!
batch_normalization_4_3414410
covtr9_3414413
covtr9_3414415!
batch_normalization_5_3414457!
batch_normalization_5_3414459!
batch_normalization_5_3414461!
batch_normalization_5_3414463
covtr10_3414466
covtr10_3414468!
batch_normalization_6_3414510!
batch_normalization_6_3414512!
batch_normalization_6_3414514!
batch_normalization_6_3414516
covtr11_3414519
covtr11_3414521!
batch_normalization_7_3414563!
batch_normalization_7_3414565!
batch_normalization_7_3414567!
batch_normalization_7_3414569
covtr14_3414572
covtr14_3414574
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallвcovtr10/StatefulPartitionedCallвcovtr11/StatefulPartitionedCallвcovtr14/StatefulPartitionedCallвcovtr4/StatefulPartitionedCallвcovtr5/StatefulPartitionedCallвcovtr6/StatefulPartitionedCallвcovtr7/StatefulPartitionedCallвcovtr8/StatefulPartitionedCallвcovtr9/StatefulPartitionedCallт
reshape/PartitionedCallPartitionedCall	gen_noise*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_34141402
reshape/PartitionedCall╞
covtr4/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr4_3414148covtr4_3414150*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr4_layer_call_and_return_conditional_losses_34128952 
covtr4/StatefulPartitionedCallЮ
leaky_re_lu/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_34141582
leaky_re_lu/PartitionedCall╟
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_3414192batch_normalization_3414194batch_normalization_3414196batch_normalization_3414198*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_34129672-
+batch_normalization/StatefulPartitionedCall┌
covtr5/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr5_3414201covtr5_3414203*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr5_layer_call_and_return_conditional_losses_34130472 
covtr5/StatefulPartitionedCallд
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_34142112
leaky_re_lu_1/PartitionedCall╫
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_3414245batch_normalization_1_3414247batch_normalization_1_3414249batch_normalization_1_3414251*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34131192/
-batch_normalization_1/StatefulPartitionedCall▄
covtr6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr6_3414254covtr6_3414256*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr6_layer_call_and_return_conditional_losses_34131992 
covtr6/StatefulPartitionedCallд
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_34142642
leaky_re_lu_2/PartitionedCall╫
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_3414298batch_normalization_2_3414300batch_normalization_2_3414302batch_normalization_2_3414304*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34132712/
-batch_normalization_2/StatefulPartitionedCall▄
covtr7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0covtr7_3414307covtr7_3414309*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr7_layer_call_and_return_conditional_losses_34133512 
covtr7/StatefulPartitionedCallд
leaky_re_lu_3/PartitionedCallPartitionedCall'covtr7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_34143172
leaky_re_lu_3/PartitionedCall╫
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0batch_normalization_3_3414351batch_normalization_3_3414353batch_normalization_3_3414355batch_normalization_3_3414357*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34134232/
-batch_normalization_3/StatefulPartitionedCall▄
covtr8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0covtr8_3414360covtr8_3414362*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr8_layer_call_and_return_conditional_losses_34135032 
covtr8/StatefulPartitionedCallд
leaky_re_lu_4/PartitionedCallPartitionedCall'covtr8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_34143702
leaky_re_lu_4/PartitionedCall╫
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0batch_normalization_4_3414404batch_normalization_4_3414406batch_normalization_4_3414408batch_normalization_4_3414410*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34135752/
-batch_normalization_4/StatefulPartitionedCall▄
covtr9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0covtr9_3414413covtr9_3414415*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr9_layer_call_and_return_conditional_losses_34136552 
covtr9/StatefulPartitionedCallд
leaky_re_lu_5/PartitionedCallPartitionedCall'covtr9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_34144232
leaky_re_lu_5/PartitionedCall╫
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0batch_normalization_5_3414457batch_normalization_5_3414459batch_normalization_5_3414461batch_normalization_5_3414463*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34137272/
-batch_normalization_5/StatefulPartitionedCallс
covtr10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0covtr10_3414466covtr10_3414468*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_covtr10_layer_call_and_return_conditional_losses_34138072!
covtr10/StatefulPartitionedCallе
leaky_re_lu_6/PartitionedCallPartitionedCall(covtr10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_34144762
leaky_re_lu_6/PartitionedCall╫
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0batch_normalization_6_3414510batch_normalization_6_3414512batch_normalization_6_3414514batch_normalization_6_3414516*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34138792/
-batch_normalization_6/StatefulPartitionedCallс
covtr11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0covtr11_3414519covtr11_3414521*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           "*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_covtr11_layer_call_and_return_conditional_losses_34139592!
covtr11/StatefulPartitionedCallе
leaky_re_lu_7/PartitionedCallPartitionedCall(covtr11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           "* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_34145292
leaky_re_lu_7/PartitionedCall╫
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0batch_normalization_7_3414563batch_normalization_7_3414565batch_normalization_7_3414567batch_normalization_7_3414569*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           "*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34140312/
-batch_normalization_7/StatefulPartitionedCallс
covtr14/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0covtr14_3414572covtr14_3414574*
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
GPU2 *0J 8В *M
fHRF
D__inference_covtr14_layer_call_and_return_conditional_losses_34141122!
covtr14/StatefulPartitionedCall└
IdentityIdentity(covtr14/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall ^covtr10/StatefulPartitionedCall ^covtr11/StatefulPartitionedCall ^covtr14/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^covtr5/StatefulPartitionedCall^covtr6/StatefulPartitionedCall^covtr7/StatefulPartitionedCall^covtr8/StatefulPartitionedCall^covtr9/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         с::::::::::::::::::::::::::::::::::::::::::::::::::2Z
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
:         с
#
_user_specified_name	gen_noise
к
d
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_3414158

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╩
п
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3413271

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
м
f
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_3414264

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ъ
Л
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_3416561

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
м
f
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_3416444

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╩
п
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3416395

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
и
Ї
%__inference_signature_wrapper_3415286
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
identityИвStatefulPartitionedCall∙
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
:         7-*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*2
config_proto" 

CPU

GPU2 *0J 8В *+
f&R$
"__inference__wrapped_model_34128572
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         7-2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         с::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
(
_output_shapes
:         с
#
_user_specified_name	gen_noise
д
к
7__inference_batch_normalization_2_layer_call_fn_3416352

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
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34132712
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ж
к
7__inference_batch_normalization_5_layer_call_fn_3416587

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
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34137582
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
а
и
5__inference_batch_normalization_layer_call_fn_3416204

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
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_34129672
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╔
~
)__inference_covtr11_layer_call_fn_3413969

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
 *A
_output_shapes/
-:+                           "*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_covtr11_layer_call_and_return_conditional_losses_34139592
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           "2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
й
E
)__inference_reshape_layer_call_fn_3416143

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
:         * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_34141402
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*'
_input_shapes
:         с:P L
(
_output_shapes
:         с
 
_user_specified_nameinputs
в
и
5__inference_batch_normalization_layer_call_fn_3416217

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
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_34129982
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ж
к
7__inference_batch_normalization_2_layer_call_fn_3416365

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
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34133022
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╟
}
(__inference_covtr4_layer_call_fn_3412905

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
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr4_layer_call_and_return_conditional_losses_34128952
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
м
f
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_3416370

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
├$
╡
C__inference_covtr8_layer_call_and_return_conditional_losses_3413503

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
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpё
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpд
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           :::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╟
}
(__inference_covtr8_layer_call_fn_3413513

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
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr8_layer_call_and_return_conditional_losses_34135032
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ж
к
7__inference_batch_normalization_3_layer_call_fn_3416439

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
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34134542
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╓С
╪
F__inference_Generator_layer_call_and_return_conditional_losses_3415608

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
identityИв"batch_normalization/AssignNewValueв$batch_normalization/AssignNewValue_1в$batch_normalization_1/AssignNewValueв&batch_normalization_1/AssignNewValue_1в$batch_normalization_2/AssignNewValueв&batch_normalization_2/AssignNewValue_1в$batch_normalization_3/AssignNewValueв&batch_normalization_3/AssignNewValue_1в$batch_normalization_4/AssignNewValueв&batch_normalization_4/AssignNewValue_1в$batch_normalization_5/AssignNewValueв&batch_normalization_5/AssignNewValue_1в$batch_normalization_6/AssignNewValueв&batch_normalization_6/AssignNewValue_1в$batch_normalization_7/AssignNewValueв&batch_normalization_7/AssignNewValue_1T
reshape/ShapeShapeinputs*
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
reshape/Reshape/shape/3ъ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeП
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:         2
reshape/Reshaped
covtr4/ShapeShapereshape/Reshape:output:0*
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
covtr4/strided_slice_1╚
&covtr4/conv2d_transpose/ReadVariableOpReadVariableOp/covtr4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr4/conv2d_transpose/ReadVariableOpН
covtr4/conv2d_transposeConv2DBackpropInputcovtr4/stack:output:0.covtr4/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
covtr4/conv2d_transposeб
covtr4/BiasAdd/ReadVariableOpReadVariableOp&covtr4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr4/BiasAdd/ReadVariableOpо
covtr4/BiasAddBiasAdd covtr4/conv2d_transpose:output:0%covtr4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
covtr4/BiasAddЕ
leaky_re_lu/LeakyRelu	LeakyRelucovtr4/BiasAdd:output:0*/
_output_shapes
:         2
leaky_re_lu/LeakyRelu░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp╢
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1э
$batch_normalization/FusedBatchNormV3FusedBatchNormV3#leaky_re_lu/LeakyRelu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2&
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
$batch_normalization/AssignNewValue_1t
covtr5/ShapeShape(batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr5/ShapeВ
covtr5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr5/strided_slice/stackЖ
covtr5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr5/strided_slice/stack_1Ж
covtr5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr5/strided_slice/stack_2М
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
covtr5/stack/3╝
covtr5/stackPackcovtr5/strided_slice:output:0covtr5/stack/1:output:0covtr5/stack/2:output:0covtr5/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr5/stackЖ
covtr5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr5/strided_slice_1/stackК
covtr5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr5/strided_slice_1/stack_1К
covtr5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr5/strided_slice_1/stack_2Ц
covtr5/strided_slice_1StridedSlicecovtr5/stack:output:0%covtr5/strided_slice_1/stack:output:0'covtr5/strided_slice_1/stack_1:output:0'covtr5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr5/strided_slice_1╚
&covtr5/conv2d_transpose/ReadVariableOpReadVariableOp/covtr5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr5/conv2d_transpose/ReadVariableOpЭ
covtr5/conv2d_transposeConv2DBackpropInputcovtr5/stack:output:0.covtr5/conv2d_transpose/ReadVariableOp:value:0(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
covtr5/conv2d_transposeб
covtr5/BiasAdd/ReadVariableOpReadVariableOp&covtr5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr5/BiasAdd/ReadVariableOpо
covtr5/BiasAddBiasAdd covtr5/conv2d_transpose:output:0%covtr5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
covtr5/BiasAddЙ
leaky_re_lu_1/LeakyRelu	LeakyRelucovtr5/BiasAdd:output:0*/
_output_shapes
:         2
leaky_re_lu_1/LeakyRelu╢
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1√
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_1/LeakyRelu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
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
&batch_normalization_1/AssignNewValue_1v
covtr6/ShapeShape*batch_normalization_1/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr6/ShapeВ
covtr6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr6/strided_slice/stackЖ
covtr6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr6/strided_slice/stack_1Ж
covtr6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr6/strided_slice/stack_2М
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
covtr6/stack/3╝
covtr6/stackPackcovtr6/strided_slice:output:0covtr6/stack/1:output:0covtr6/stack/2:output:0covtr6/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr6/stackЖ
covtr6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr6/strided_slice_1/stackК
covtr6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr6/strided_slice_1/stack_1К
covtr6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr6/strided_slice_1/stack_2Ц
covtr6/strided_slice_1StridedSlicecovtr6/stack:output:0%covtr6/strided_slice_1/stack:output:0'covtr6/strided_slice_1/stack_1:output:0'covtr6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr6/strided_slice_1╚
&covtr6/conv2d_transpose/ReadVariableOpReadVariableOp/covtr6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr6/conv2d_transpose/ReadVariableOpЯ
covtr6/conv2d_transposeConv2DBackpropInputcovtr6/stack:output:0.covtr6/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
covtr6/conv2d_transposeб
covtr6/BiasAdd/ReadVariableOpReadVariableOp&covtr6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr6/BiasAdd/ReadVariableOpо
covtr6/BiasAddBiasAdd covtr6/conv2d_transpose:output:0%covtr6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
covtr6/BiasAddЙ
leaky_re_lu_2/LeakyRelu	LeakyRelucovtr6/BiasAdd:output:0*/
_output_shapes
:         2
leaky_re_lu_2/LeakyRelu╢
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_2/ReadVariableOp╝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1щ
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1√
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_2/LeakyRelu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
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
&batch_normalization_2/AssignNewValue_1v
covtr7/ShapeShape*batch_normalization_2/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr7/ShapeВ
covtr7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr7/strided_slice/stackЖ
covtr7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr7/strided_slice/stack_1Ж
covtr7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr7/strided_slice/stack_2М
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
covtr7/stack/3╝
covtr7/stackPackcovtr7/strided_slice:output:0covtr7/stack/1:output:0covtr7/stack/2:output:0covtr7/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr7/stackЖ
covtr7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr7/strided_slice_1/stackК
covtr7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr7/strided_slice_1/stack_1К
covtr7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr7/strided_slice_1/stack_2Ц
covtr7/strided_slice_1StridedSlicecovtr7/stack:output:0%covtr7/strided_slice_1/stack:output:0'covtr7/strided_slice_1/stack_1:output:0'covtr7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr7/strided_slice_1╚
&covtr7/conv2d_transpose/ReadVariableOpReadVariableOp/covtr7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr7/conv2d_transpose/ReadVariableOpЯ
covtr7/conv2d_transposeConv2DBackpropInputcovtr7/stack:output:0.covtr7/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
covtr7/conv2d_transposeб
covtr7/BiasAdd/ReadVariableOpReadVariableOp&covtr7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr7/BiasAdd/ReadVariableOpо
covtr7/BiasAddBiasAdd covtr7/conv2d_transpose:output:0%covtr7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
covtr7/BiasAddЙ
leaky_re_lu_3/LeakyRelu	LeakyRelucovtr7/BiasAdd:output:0*/
_output_shapes
:         2
leaky_re_lu_3/LeakyRelu╢
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_3/ReadVariableOp╝
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_3/ReadVariableOp_1щ
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1√
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_3/LeakyRelu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_3/FusedBatchNormV3Г
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValueС
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
covtr8/ShapeВ
covtr8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr8/strided_slice/stackЖ
covtr8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr8/strided_slice/stack_1Ж
covtr8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr8/strided_slice/stack_2М
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
covtr8/stack/3╝
covtr8/stackPackcovtr8/strided_slice:output:0covtr8/stack/1:output:0covtr8/stack/2:output:0covtr8/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr8/stackЖ
covtr8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr8/strided_slice_1/stackК
covtr8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr8/strided_slice_1/stack_1К
covtr8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr8/strided_slice_1/stack_2Ц
covtr8/strided_slice_1StridedSlicecovtr8/stack:output:0%covtr8/strided_slice_1/stack:output:0'covtr8/strided_slice_1/stack_1:output:0'covtr8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr8/strided_slice_1╚
&covtr8/conv2d_transpose/ReadVariableOpReadVariableOp/covtr8_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr8/conv2d_transpose/ReadVariableOpЯ
covtr8/conv2d_transposeConv2DBackpropInputcovtr8/stack:output:0.covtr8/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
covtr8/conv2d_transposeб
covtr8/BiasAdd/ReadVariableOpReadVariableOp&covtr8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr8/BiasAdd/ReadVariableOpо
covtr8/BiasAddBiasAdd covtr8/conv2d_transpose:output:0%covtr8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
covtr8/BiasAddЙ
leaky_re_lu_4/LeakyRelu	LeakyRelucovtr8/BiasAdd:output:0*/
_output_shapes
:         2
leaky_re_lu_4/LeakyRelu╢
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_4/ReadVariableOp╝
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1щ
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1√
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_4/LeakyRelu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_4/FusedBatchNormV3Г
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValueС
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
covtr9/ShapeВ
covtr9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr9/strided_slice/stackЖ
covtr9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr9/strided_slice/stack_1Ж
covtr9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr9/strided_slice/stack_2М
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
covtr9/stack/3╝
covtr9/stackPackcovtr9/strided_slice:output:0covtr9/stack/1:output:0covtr9/stack/2:output:0covtr9/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr9/stackЖ
covtr9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr9/strided_slice_1/stackК
covtr9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr9/strided_slice_1/stack_1К
covtr9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr9/strided_slice_1/stack_2Ц
covtr9/strided_slice_1StridedSlicecovtr9/stack:output:0%covtr9/strided_slice_1/stack:output:0'covtr9/strided_slice_1/stack_1:output:0'covtr9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr9/strided_slice_1╚
&covtr9/conv2d_transpose/ReadVariableOpReadVariableOp/covtr9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr9/conv2d_transpose/ReadVariableOpЯ
covtr9/conv2d_transposeConv2DBackpropInputcovtr9/stack:output:0.covtr9/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         $*
paddingVALID*
strides
2
covtr9/conv2d_transposeб
covtr9/BiasAdd/ReadVariableOpReadVariableOp&covtr9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr9/BiasAdd/ReadVariableOpо
covtr9/BiasAddBiasAdd covtr9/conv2d_transpose:output:0%covtr9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         $2
covtr9/BiasAddЙ
leaky_re_lu_5/LeakyRelu	LeakyRelucovtr9/BiasAdd:output:0*/
_output_shapes
:         $2
leaky_re_lu_5/LeakyRelu╢
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_5/ReadVariableOp╝
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_5/ReadVariableOp_1щ
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1√
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_5/LeakyRelu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         $:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_5/FusedBatchNormV3Г
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValueС
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
covtr10/ShapeД
covtr10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr10/strided_slice/stackИ
covtr10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr10/strided_slice/stack_1И
covtr10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr10/strided_slice/stack_2Т
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
covtr10/stack/3┬
covtr10/stackPackcovtr10/strided_slice:output:0covtr10/stack/1:output:0covtr10/stack/2:output:0covtr10/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr10/stackИ
covtr10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr10/strided_slice_1/stackМ
covtr10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr10/strided_slice_1/stack_1М
covtr10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr10/strided_slice_1/stack_2Ь
covtr10/strided_slice_1StridedSlicecovtr10/stack:output:0&covtr10/strided_slice_1/stack:output:0(covtr10/strided_slice_1/stack_1:output:0(covtr10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr10/strided_slice_1╦
'covtr10/conv2d_transpose/ReadVariableOpReadVariableOp0covtr10_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02)
'covtr10/conv2d_transpose/ReadVariableOpг
covtr10/conv2d_transposeConv2DBackpropInputcovtr10/stack:output:0/covtr10/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         +!*
paddingVALID*
strides
2
covtr10/conv2d_transposeд
covtr10/BiasAdd/ReadVariableOpReadVariableOp'covtr10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
covtr10/BiasAdd/ReadVariableOp▓
covtr10/BiasAddBiasAdd!covtr10/conv2d_transpose:output:0&covtr10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         +!2
covtr10/BiasAddК
leaky_re_lu_6/LeakyRelu	LeakyRelucovtr10/BiasAdd:output:0*/
_output_shapes
:         +!2
leaky_re_lu_6/LeakyRelu╢
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_6/ReadVariableOp╝
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1щ
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1√
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_6/LeakyRelu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         +!:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_6/FusedBatchNormV3Г
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValueС
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
covtr11/ShapeД
covtr11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr11/strided_slice/stackИ
covtr11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr11/strided_slice/stack_1И
covtr11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr11/strided_slice/stack_2Т
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
covtr11/stack/3┬
covtr11/stackPackcovtr11/strided_slice:output:0covtr11/stack/1:output:0covtr11/stack/2:output:0covtr11/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr11/stackИ
covtr11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr11/strided_slice_1/stackМ
covtr11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr11/strided_slice_1/stack_1М
covtr11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr11/strided_slice_1/stack_2Ь
covtr11/strided_slice_1StridedSlicecovtr11/stack:output:0&covtr11/strided_slice_1/stack:output:0(covtr11/strided_slice_1/stack_1:output:0(covtr11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr11/strided_slice_1╦
'covtr11/conv2d_transpose/ReadVariableOpReadVariableOp0covtr11_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		"*
dtype02)
'covtr11/conv2d_transpose/ReadVariableOpг
covtr11/conv2d_transposeConv2DBackpropInputcovtr11/stack:output:0/covtr11/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         3)"*
paddingVALID*
strides
2
covtr11/conv2d_transposeд
covtr11/BiasAdd/ReadVariableOpReadVariableOp'covtr11_biasadd_readvariableop_resource*
_output_shapes
:"*
dtype02 
covtr11/BiasAdd/ReadVariableOp▓
covtr11/BiasAddBiasAdd!covtr11/conv2d_transpose:output:0&covtr11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         3)"2
covtr11/BiasAddК
leaky_re_lu_7/LeakyRelu	LeakyRelucovtr11/BiasAdd:output:0*/
_output_shapes
:         3)"2
leaky_re_lu_7/LeakyRelu╢
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:"*
dtype02&
$batch_normalization_7/ReadVariableOp╝
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:"*
dtype02(
&batch_normalization_7/ReadVariableOp_1щ
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:"*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:"*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1√
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_7/LeakyRelu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         3)":":":":":*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_7/FusedBatchNormV3Г
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValueС
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
covtr14/ShapeД
covtr14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr14/strided_slice/stackИ
covtr14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr14/strided_slice/stack_1И
covtr14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr14/strided_slice/stack_2Т
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
covtr14/stack/3┬
covtr14/stackPackcovtr14/strided_slice:output:0covtr14/stack/1:output:0covtr14/stack/2:output:0covtr14/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr14/stackИ
covtr14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr14/strided_slice_1/stackМ
covtr14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr14/strided_slice_1/stack_1М
covtr14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr14/strided_slice_1/stack_2Ь
covtr14/strided_slice_1StridedSlicecovtr14/stack:output:0&covtr14/strided_slice_1/stack:output:0(covtr14/strided_slice_1/stack_1:output:0(covtr14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr14/strided_slice_1╦
'covtr14/conv2d_transpose/ReadVariableOpReadVariableOp0covtr14_conv2d_transpose_readvariableop_resource*&
_output_shapes
:"*
dtype02)
'covtr14/conv2d_transpose/ReadVariableOpг
covtr14/conv2d_transposeConv2DBackpropInputcovtr14/stack:output:0/covtr14/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         7-*
paddingVALID*
strides
2
covtr14/conv2d_transposeд
covtr14/BiasAdd/ReadVariableOpReadVariableOp'covtr14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
covtr14/BiasAdd/ReadVariableOp▓
covtr14/BiasAddBiasAdd!covtr14/conv2d_transpose:output:0&covtr14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         7-2
covtr14/BiasAddБ
covtr14/SigmoidSigmoidcovtr14/BiasAdd:output:0*
T0*/
_output_shapes
:         7-2
covtr14/Sigmoidы
IdentityIdentitycovtr14/Sigmoid:y:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_1*
T0*/
_output_shapes
:         7-2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         с::::::::::::::::::::::::::::::::::::::::::::::::::2H
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
:         с
 
_user_specified_nameinputs
М
K
/__inference_leaky_re_lu_2_layer_call_fn_3416301

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
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_34142642
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╩
п
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3414031

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:"*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:"*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:"*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:"*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           ":":":":":*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
-:+                           "2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           "::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           "
 
_user_specified_nameinputs
▌
ў
+__inference_Generator_layer_call_fn_3416019

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
identityИвStatefulPartitionedCallЬ
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
-:+                           *D
_read_only_resource_inputs&
$"	
 !"%&'(+,-.12*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_Generator_layer_call_and_return_conditional_losses_34148412
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         с::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         с
 
_user_specified_nameinputs
╩
п
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3413879

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ъ
Л
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3416265

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╦Е
п
"__inference__wrapped_model_3412857
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
identityИk
Generator/reshape/ShapeShape	gen_noise*
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
value	B :2#
!Generator/reshape/Reshape/shape/1И
!Generator/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
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
Generator/reshape/Reshape/shape░
Generator/reshape/ReshapeReshape	gen_noise(Generator/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:         2
Generator/reshape/ReshapeВ
Generator/covtr4/ShapeShape"Generator/reshape/Reshape:output:0*
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
 Generator/covtr4/strided_slice_1ц
0Generator/covtr4/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0Generator/covtr4/conv2d_transpose/ReadVariableOp┐
!Generator/covtr4/conv2d_transposeConv2DBackpropInputGenerator/covtr4/stack:output:08Generator/covtr4/conv2d_transpose/ReadVariableOp:value:0"Generator/reshape/Reshape:output:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2#
!Generator/covtr4/conv2d_transpose┐
'Generator/covtr4/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Generator/covtr4/BiasAdd/ReadVariableOp╓
Generator/covtr4/BiasAddBiasAdd*Generator/covtr4/conv2d_transpose:output:0/Generator/covtr4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
Generator/covtr4/BiasAddг
Generator/leaky_re_lu/LeakyRelu	LeakyRelu!Generator/covtr4/BiasAdd:output:0*/
_output_shapes
:         2!
Generator/leaky_re_lu/LeakyRelu╬
,Generator/batch_normalization/ReadVariableOpReadVariableOp5generator_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02.
,Generator/batch_normalization/ReadVariableOp╘
.Generator/batch_normalization/ReadVariableOp_1ReadVariableOp7generator_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype020
.Generator/batch_normalization/ReadVariableOp_1Б
=Generator/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpFgenerator_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=Generator/batch_normalization/FusedBatchNormV3/ReadVariableOpЗ
?Generator/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHgenerator_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?Generator/batch_normalization/FusedBatchNormV3/ReadVariableOp_1е
.Generator/batch_normalization/FusedBatchNormV3FusedBatchNormV3-Generator/leaky_re_lu/LeakyRelu:activations:04Generator/batch_normalization/ReadVariableOp:value:06Generator/batch_normalization/ReadVariableOp_1:value:0EGenerator/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0GGenerator/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
is_training( 20
.Generator/batch_normalization/FusedBatchNormV3Т
Generator/covtr5/ShapeShape2Generator/batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr5/ShapeЦ
$Generator/covtr5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr5/strided_slice/stackЪ
&Generator/covtr5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr5/strided_slice/stack_1Ъ
&Generator/covtr5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr5/strided_slice/stack_2╚
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
Generator/covtr5/stack/3°
Generator/covtr5/stackPack'Generator/covtr5/strided_slice:output:0!Generator/covtr5/stack/1:output:0!Generator/covtr5/stack/2:output:0!Generator/covtr5/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr5/stackЪ
&Generator/covtr5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr5/strided_slice_1/stackЮ
(Generator/covtr5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr5/strided_slice_1/stack_1Ю
(Generator/covtr5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr5/strided_slice_1/stack_2╥
 Generator/covtr5/strided_slice_1StridedSliceGenerator/covtr5/stack:output:0/Generator/covtr5/strided_slice_1/stack:output:01Generator/covtr5/strided_slice_1/stack_1:output:01Generator/covtr5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr5/strided_slice_1ц
0Generator/covtr5/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0Generator/covtr5/conv2d_transpose/ReadVariableOp╧
!Generator/covtr5/conv2d_transposeConv2DBackpropInputGenerator/covtr5/stack:output:08Generator/covtr5/conv2d_transpose/ReadVariableOp:value:02Generator/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2#
!Generator/covtr5/conv2d_transpose┐
'Generator/covtr5/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Generator/covtr5/BiasAdd/ReadVariableOp╓
Generator/covtr5/BiasAddBiasAdd*Generator/covtr5/conv2d_transpose:output:0/Generator/covtr5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
Generator/covtr5/BiasAddз
!Generator/leaky_re_lu_1/LeakyRelu	LeakyRelu!Generator/covtr5/BiasAdd:output:0*/
_output_shapes
:         2#
!Generator/leaky_re_lu_1/LeakyRelu╘
.Generator/batch_normalization_1/ReadVariableOpReadVariableOp7generator_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype020
.Generator/batch_normalization_1/ReadVariableOp┌
0Generator/batch_normalization_1/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype022
0Generator/batch_normalization_1/ReadVariableOp_1З
?Generator/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?Generator/batch_normalization_1/FusedBatchNormV3/ReadVariableOpН
AGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
AGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1│
0Generator/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_1/LeakyRelu:activations:06Generator/batch_normalization_1/ReadVariableOp:value:08Generator/batch_normalization_1/ReadVariableOp_1:value:0GGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
is_training( 22
0Generator/batch_normalization_1/FusedBatchNormV3Ф
Generator/covtr6/ShapeShape4Generator/batch_normalization_1/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr6/ShapeЦ
$Generator/covtr6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr6/strided_slice/stackЪ
&Generator/covtr6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr6/strided_slice/stack_1Ъ
&Generator/covtr6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr6/strided_slice/stack_2╚
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
Generator/covtr6/stack/3°
Generator/covtr6/stackPack'Generator/covtr6/strided_slice:output:0!Generator/covtr6/stack/1:output:0!Generator/covtr6/stack/2:output:0!Generator/covtr6/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr6/stackЪ
&Generator/covtr6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr6/strided_slice_1/stackЮ
(Generator/covtr6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr6/strided_slice_1/stack_1Ю
(Generator/covtr6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr6/strided_slice_1/stack_2╥
 Generator/covtr6/strided_slice_1StridedSliceGenerator/covtr6/stack:output:0/Generator/covtr6/strided_slice_1/stack:output:01Generator/covtr6/strided_slice_1/stack_1:output:01Generator/covtr6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr6/strided_slice_1ц
0Generator/covtr6/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0Generator/covtr6/conv2d_transpose/ReadVariableOp╤
!Generator/covtr6/conv2d_transposeConv2DBackpropInputGenerator/covtr6/stack:output:08Generator/covtr6/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2#
!Generator/covtr6/conv2d_transpose┐
'Generator/covtr6/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Generator/covtr6/BiasAdd/ReadVariableOp╓
Generator/covtr6/BiasAddBiasAdd*Generator/covtr6/conv2d_transpose:output:0/Generator/covtr6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
Generator/covtr6/BiasAddз
!Generator/leaky_re_lu_2/LeakyRelu	LeakyRelu!Generator/covtr6/BiasAdd:output:0*/
_output_shapes
:         2#
!Generator/leaky_re_lu_2/LeakyRelu╘
.Generator/batch_normalization_2/ReadVariableOpReadVariableOp7generator_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype020
.Generator/batch_normalization_2/ReadVariableOp┌
0Generator/batch_normalization_2/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype022
0Generator/batch_normalization_2/ReadVariableOp_1З
?Generator/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?Generator/batch_normalization_2/FusedBatchNormV3/ReadVariableOpН
AGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
AGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1│
0Generator/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_2/LeakyRelu:activations:06Generator/batch_normalization_2/ReadVariableOp:value:08Generator/batch_normalization_2/ReadVariableOp_1:value:0GGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
is_training( 22
0Generator/batch_normalization_2/FusedBatchNormV3Ф
Generator/covtr7/ShapeShape4Generator/batch_normalization_2/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr7/ShapeЦ
$Generator/covtr7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr7/strided_slice/stackЪ
&Generator/covtr7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr7/strided_slice/stack_1Ъ
&Generator/covtr7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr7/strided_slice/stack_2╚
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
Generator/covtr7/stack/3°
Generator/covtr7/stackPack'Generator/covtr7/strided_slice:output:0!Generator/covtr7/stack/1:output:0!Generator/covtr7/stack/2:output:0!Generator/covtr7/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr7/stackЪ
&Generator/covtr7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr7/strided_slice_1/stackЮ
(Generator/covtr7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr7/strided_slice_1/stack_1Ю
(Generator/covtr7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr7/strided_slice_1/stack_2╥
 Generator/covtr7/strided_slice_1StridedSliceGenerator/covtr7/stack:output:0/Generator/covtr7/strided_slice_1/stack:output:01Generator/covtr7/strided_slice_1/stack_1:output:01Generator/covtr7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr7/strided_slice_1ц
0Generator/covtr7/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0Generator/covtr7/conv2d_transpose/ReadVariableOp╤
!Generator/covtr7/conv2d_transposeConv2DBackpropInputGenerator/covtr7/stack:output:08Generator/covtr7/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2#
!Generator/covtr7/conv2d_transpose┐
'Generator/covtr7/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Generator/covtr7/BiasAdd/ReadVariableOp╓
Generator/covtr7/BiasAddBiasAdd*Generator/covtr7/conv2d_transpose:output:0/Generator/covtr7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
Generator/covtr7/BiasAddз
!Generator/leaky_re_lu_3/LeakyRelu	LeakyRelu!Generator/covtr7/BiasAdd:output:0*/
_output_shapes
:         2#
!Generator/leaky_re_lu_3/LeakyRelu╘
.Generator/batch_normalization_3/ReadVariableOpReadVariableOp7generator_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype020
.Generator/batch_normalization_3/ReadVariableOp┌
0Generator/batch_normalization_3/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype022
0Generator/batch_normalization_3/ReadVariableOp_1З
?Generator/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?Generator/batch_normalization_3/FusedBatchNormV3/ReadVariableOpН
AGenerator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
AGenerator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1│
0Generator/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_3/LeakyRelu:activations:06Generator/batch_normalization_3/ReadVariableOp:value:08Generator/batch_normalization_3/ReadVariableOp_1:value:0GGenerator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
is_training( 22
0Generator/batch_normalization_3/FusedBatchNormV3Ф
Generator/covtr8/ShapeShape4Generator/batch_normalization_3/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr8/ShapeЦ
$Generator/covtr8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr8/strided_slice/stackЪ
&Generator/covtr8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr8/strided_slice/stack_1Ъ
&Generator/covtr8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr8/strided_slice/stack_2╚
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
Generator/covtr8/stack/3°
Generator/covtr8/stackPack'Generator/covtr8/strided_slice:output:0!Generator/covtr8/stack/1:output:0!Generator/covtr8/stack/2:output:0!Generator/covtr8/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr8/stackЪ
&Generator/covtr8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr8/strided_slice_1/stackЮ
(Generator/covtr8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr8/strided_slice_1/stack_1Ю
(Generator/covtr8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr8/strided_slice_1/stack_2╥
 Generator/covtr8/strided_slice_1StridedSliceGenerator/covtr8/stack:output:0/Generator/covtr8/strided_slice_1/stack:output:01Generator/covtr8/strided_slice_1/stack_1:output:01Generator/covtr8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr8/strided_slice_1ц
0Generator/covtr8/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr8_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0Generator/covtr8/conv2d_transpose/ReadVariableOp╤
!Generator/covtr8/conv2d_transposeConv2DBackpropInputGenerator/covtr8/stack:output:08Generator/covtr8/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2#
!Generator/covtr8/conv2d_transpose┐
'Generator/covtr8/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Generator/covtr8/BiasAdd/ReadVariableOp╓
Generator/covtr8/BiasAddBiasAdd*Generator/covtr8/conv2d_transpose:output:0/Generator/covtr8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
Generator/covtr8/BiasAddз
!Generator/leaky_re_lu_4/LeakyRelu	LeakyRelu!Generator/covtr8/BiasAdd:output:0*/
_output_shapes
:         2#
!Generator/leaky_re_lu_4/LeakyRelu╘
.Generator/batch_normalization_4/ReadVariableOpReadVariableOp7generator_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype020
.Generator/batch_normalization_4/ReadVariableOp┌
0Generator/batch_normalization_4/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype022
0Generator/batch_normalization_4/ReadVariableOp_1З
?Generator/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?Generator/batch_normalization_4/FusedBatchNormV3/ReadVariableOpН
AGenerator/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
AGenerator/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1│
0Generator/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_4/LeakyRelu:activations:06Generator/batch_normalization_4/ReadVariableOp:value:08Generator/batch_normalization_4/ReadVariableOp_1:value:0GGenerator/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
is_training( 22
0Generator/batch_normalization_4/FusedBatchNormV3Ф
Generator/covtr9/ShapeShape4Generator/batch_normalization_4/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr9/ShapeЦ
$Generator/covtr9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Generator/covtr9/strided_slice/stackЪ
&Generator/covtr9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr9/strided_slice/stack_1Ъ
&Generator/covtr9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Generator/covtr9/strided_slice/stack_2╚
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
Generator/covtr9/stack/3°
Generator/covtr9/stackPack'Generator/covtr9/strided_slice:output:0!Generator/covtr9/stack/1:output:0!Generator/covtr9/stack/2:output:0!Generator/covtr9/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr9/stackЪ
&Generator/covtr9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Generator/covtr9/strided_slice_1/stackЮ
(Generator/covtr9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr9/strided_slice_1/stack_1Ю
(Generator/covtr9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Generator/covtr9/strided_slice_1/stack_2╥
 Generator/covtr9/strided_slice_1StridedSliceGenerator/covtr9/stack:output:0/Generator/covtr9/strided_slice_1/stack:output:01Generator/covtr9/strided_slice_1/stack_1:output:01Generator/covtr9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Generator/covtr9/strided_slice_1ц
0Generator/covtr9/conv2d_transpose/ReadVariableOpReadVariableOp9generator_covtr9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0Generator/covtr9/conv2d_transpose/ReadVariableOp╤
!Generator/covtr9/conv2d_transposeConv2DBackpropInputGenerator/covtr9/stack:output:08Generator/covtr9/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         $*
paddingVALID*
strides
2#
!Generator/covtr9/conv2d_transpose┐
'Generator/covtr9/BiasAdd/ReadVariableOpReadVariableOp0generator_covtr9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Generator/covtr9/BiasAdd/ReadVariableOp╓
Generator/covtr9/BiasAddBiasAdd*Generator/covtr9/conv2d_transpose:output:0/Generator/covtr9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         $2
Generator/covtr9/BiasAddз
!Generator/leaky_re_lu_5/LeakyRelu	LeakyRelu!Generator/covtr9/BiasAdd:output:0*/
_output_shapes
:         $2#
!Generator/leaky_re_lu_5/LeakyRelu╘
.Generator/batch_normalization_5/ReadVariableOpReadVariableOp7generator_batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype020
.Generator/batch_normalization_5/ReadVariableOp┌
0Generator/batch_normalization_5/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype022
0Generator/batch_normalization_5/ReadVariableOp_1З
?Generator/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?Generator/batch_normalization_5/FusedBatchNormV3/ReadVariableOpН
AGenerator/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
AGenerator/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1│
0Generator/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_5/LeakyRelu:activations:06Generator/batch_normalization_5/ReadVariableOp:value:08Generator/batch_normalization_5/ReadVariableOp_1:value:0GGenerator/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         $:::::*
epsilon%oГ:*
is_training( 22
0Generator/batch_normalization_5/FusedBatchNormV3Ц
Generator/covtr10/ShapeShape4Generator/batch_normalization_5/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr10/ShapeШ
%Generator/covtr10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Generator/covtr10/strided_slice/stackЬ
'Generator/covtr10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/covtr10/strided_slice/stack_1Ь
'Generator/covtr10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/covtr10/strided_slice/stack_2╬
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
Generator/covtr10/stack/3■
Generator/covtr10/stackPack(Generator/covtr10/strided_slice:output:0"Generator/covtr10/stack/1:output:0"Generator/covtr10/stack/2:output:0"Generator/covtr10/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr10/stackЬ
'Generator/covtr10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'Generator/covtr10/strided_slice_1/stackа
)Generator/covtr10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)Generator/covtr10/strided_slice_1/stack_1а
)Generator/covtr10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)Generator/covtr10/strided_slice_1/stack_2╪
!Generator/covtr10/strided_slice_1StridedSlice Generator/covtr10/stack:output:00Generator/covtr10/strided_slice_1/stack:output:02Generator/covtr10/strided_slice_1/stack_1:output:02Generator/covtr10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!Generator/covtr10/strided_slice_1щ
1Generator/covtr10/conv2d_transpose/ReadVariableOpReadVariableOp:generator_covtr10_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype023
1Generator/covtr10/conv2d_transpose/ReadVariableOp╒
"Generator/covtr10/conv2d_transposeConv2DBackpropInput Generator/covtr10/stack:output:09Generator/covtr10/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         +!*
paddingVALID*
strides
2$
"Generator/covtr10/conv2d_transpose┬
(Generator/covtr10/BiasAdd/ReadVariableOpReadVariableOp1generator_covtr10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Generator/covtr10/BiasAdd/ReadVariableOp┌
Generator/covtr10/BiasAddBiasAdd+Generator/covtr10/conv2d_transpose:output:00Generator/covtr10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         +!2
Generator/covtr10/BiasAddи
!Generator/leaky_re_lu_6/LeakyRelu	LeakyRelu"Generator/covtr10/BiasAdd:output:0*/
_output_shapes
:         +!2#
!Generator/leaky_re_lu_6/LeakyRelu╘
.Generator/batch_normalization_6/ReadVariableOpReadVariableOp7generator_batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype020
.Generator/batch_normalization_6/ReadVariableOp┌
0Generator/batch_normalization_6/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype022
0Generator/batch_normalization_6/ReadVariableOp_1З
?Generator/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?Generator/batch_normalization_6/FusedBatchNormV3/ReadVariableOpН
AGenerator/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
AGenerator/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1│
0Generator/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_6/LeakyRelu:activations:06Generator/batch_normalization_6/ReadVariableOp:value:08Generator/batch_normalization_6/ReadVariableOp_1:value:0GGenerator/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         +!:::::*
epsilon%oГ:*
is_training( 22
0Generator/batch_normalization_6/FusedBatchNormV3Ц
Generator/covtr11/ShapeShape4Generator/batch_normalization_6/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr11/ShapeШ
%Generator/covtr11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Generator/covtr11/strided_slice/stackЬ
'Generator/covtr11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/covtr11/strided_slice/stack_1Ь
'Generator/covtr11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/covtr11/strided_slice/stack_2╬
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
Generator/covtr11/stack/3■
Generator/covtr11/stackPack(Generator/covtr11/strided_slice:output:0"Generator/covtr11/stack/1:output:0"Generator/covtr11/stack/2:output:0"Generator/covtr11/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr11/stackЬ
'Generator/covtr11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'Generator/covtr11/strided_slice_1/stackа
)Generator/covtr11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)Generator/covtr11/strided_slice_1/stack_1а
)Generator/covtr11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)Generator/covtr11/strided_slice_1/stack_2╪
!Generator/covtr11/strided_slice_1StridedSlice Generator/covtr11/stack:output:00Generator/covtr11/strided_slice_1/stack:output:02Generator/covtr11/strided_slice_1/stack_1:output:02Generator/covtr11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!Generator/covtr11/strided_slice_1щ
1Generator/covtr11/conv2d_transpose/ReadVariableOpReadVariableOp:generator_covtr11_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		"*
dtype023
1Generator/covtr11/conv2d_transpose/ReadVariableOp╒
"Generator/covtr11/conv2d_transposeConv2DBackpropInput Generator/covtr11/stack:output:09Generator/covtr11/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         3)"*
paddingVALID*
strides
2$
"Generator/covtr11/conv2d_transpose┬
(Generator/covtr11/BiasAdd/ReadVariableOpReadVariableOp1generator_covtr11_biasadd_readvariableop_resource*
_output_shapes
:"*
dtype02*
(Generator/covtr11/BiasAdd/ReadVariableOp┌
Generator/covtr11/BiasAddBiasAdd+Generator/covtr11/conv2d_transpose:output:00Generator/covtr11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         3)"2
Generator/covtr11/BiasAddи
!Generator/leaky_re_lu_7/LeakyRelu	LeakyRelu"Generator/covtr11/BiasAdd:output:0*/
_output_shapes
:         3)"2#
!Generator/leaky_re_lu_7/LeakyRelu╘
.Generator/batch_normalization_7/ReadVariableOpReadVariableOp7generator_batch_normalization_7_readvariableop_resource*
_output_shapes
:"*
dtype020
.Generator/batch_normalization_7/ReadVariableOp┌
0Generator/batch_normalization_7/ReadVariableOp_1ReadVariableOp9generator_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:"*
dtype022
0Generator/batch_normalization_7/ReadVariableOp_1З
?Generator/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpHgenerator_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:"*
dtype02A
?Generator/batch_normalization_7/FusedBatchNormV3/ReadVariableOpН
AGenerator/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJgenerator_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:"*
dtype02C
AGenerator/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1│
0Generator/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3/Generator/leaky_re_lu_7/LeakyRelu:activations:06Generator/batch_normalization_7/ReadVariableOp:value:08Generator/batch_normalization_7/ReadVariableOp_1:value:0GGenerator/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0IGenerator/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         3)":":":":":*
epsilon%oГ:*
is_training( 22
0Generator/batch_normalization_7/FusedBatchNormV3Ц
Generator/covtr14/ShapeShape4Generator/batch_normalization_7/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/covtr14/ShapeШ
%Generator/covtr14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Generator/covtr14/strided_slice/stackЬ
'Generator/covtr14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/covtr14/strided_slice/stack_1Ь
'Generator/covtr14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Generator/covtr14/strided_slice/stack_2╬
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
Generator/covtr14/stack/3■
Generator/covtr14/stackPack(Generator/covtr14/strided_slice:output:0"Generator/covtr14/stack/1:output:0"Generator/covtr14/stack/2:output:0"Generator/covtr14/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/covtr14/stackЬ
'Generator/covtr14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'Generator/covtr14/strided_slice_1/stackа
)Generator/covtr14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)Generator/covtr14/strided_slice_1/stack_1а
)Generator/covtr14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)Generator/covtr14/strided_slice_1/stack_2╪
!Generator/covtr14/strided_slice_1StridedSlice Generator/covtr14/stack:output:00Generator/covtr14/strided_slice_1/stack:output:02Generator/covtr14/strided_slice_1/stack_1:output:02Generator/covtr14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!Generator/covtr14/strided_slice_1щ
1Generator/covtr14/conv2d_transpose/ReadVariableOpReadVariableOp:generator_covtr14_conv2d_transpose_readvariableop_resource*&
_output_shapes
:"*
dtype023
1Generator/covtr14/conv2d_transpose/ReadVariableOp╒
"Generator/covtr14/conv2d_transposeConv2DBackpropInput Generator/covtr14/stack:output:09Generator/covtr14/conv2d_transpose/ReadVariableOp:value:04Generator/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         7-*
paddingVALID*
strides
2$
"Generator/covtr14/conv2d_transpose┬
(Generator/covtr14/BiasAdd/ReadVariableOpReadVariableOp1generator_covtr14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Generator/covtr14/BiasAdd/ReadVariableOp┌
Generator/covtr14/BiasAddBiasAdd+Generator/covtr14/conv2d_transpose:output:00Generator/covtr14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         7-2
Generator/covtr14/BiasAddЯ
Generator/covtr14/SigmoidSigmoid"Generator/covtr14/BiasAdd:output:0*
T0*/
_output_shapes
:         7-2
Generator/covtr14/Sigmoidy
IdentityIdentityGenerator/covtr14/Sigmoid:y:0*
T0*/
_output_shapes
:         7-2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         с:::::::::::::::::::::::::::::::::::::::::::::::::::S O
(
_output_shapes
:         с
#
_user_specified_name	gen_noise
э
ў
+__inference_Generator_layer_call_fn_3416124

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
identityИвStatefulPartitionedCallм
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
-:+                           *T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_Generator_layer_call_and_return_conditional_losses_34150762
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         с::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         с
 
_user_specified_nameinputs
м
f
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_3414370

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ъ
Л
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3416339

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╟
}
(__inference_covtr7_layer_call_fn_3413361

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
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr7_layer_call_and_return_conditional_losses_34133512
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ж
к
7__inference_batch_normalization_4_layer_call_fn_3416513

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
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34136062
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
д
к
7__inference_batch_normalization_5_layer_call_fn_3416574

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
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34137272
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ъ
Л
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_3413606

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╝%
╢
D__inference_covtr14_layer_call_and_return_conditional_losses_3414112

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
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:"*
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
identityIdentity:output:0*H
_input_shapes7
5:+                           ":::i e
A
_output_shapes/
-:+                           "
 
_user_specified_nameinputs
Ъ
Л
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3413302

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
├$
╡
C__inference_covtr7_layer_call_and_return_conditional_losses_3413351

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
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpё
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpд
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           :::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
м
f
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_3416592

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ж
к
7__inference_batch_normalization_6_layer_call_fn_3416661

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
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34139102
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
─$
╢
D__inference_covtr11_layer_call_and_return_conditional_losses_3413959

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
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:		"*
dtype02!
conv2d_transpose/ReadVariableOpё
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           "*
paddingVALID*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:"*
dtype02
BiasAdd/ReadVariableOpд
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           "2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           "2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           :::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
М
K
/__inference_leaky_re_lu_5_layer_call_fn_3416523

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
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_34144232
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
д
к
7__inference_batch_normalization_1_layer_call_fn_3416278

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
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34131192
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ц
·
+__inference_Generator_layer_call_fn_3414944
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
identityИвStatefulPartitionedCallЯ
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
-:+                           *D
_read_only_resource_inputs&
$"	
 !"%&'(+,-.12*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_Generator_layer_call_and_return_conditional_losses_34148412
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         с::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
(
_output_shapes
:         с
#
_user_specified_name	gen_noise
ИК
╒
F__inference_Generator_layer_call_and_return_conditional_losses_3415076

inputs
covtr4_3414950
covtr4_3414952
batch_normalization_3414956
batch_normalization_3414958
batch_normalization_3414960
batch_normalization_3414962
covtr5_3414965
covtr5_3414967!
batch_normalization_1_3414971!
batch_normalization_1_3414973!
batch_normalization_1_3414975!
batch_normalization_1_3414977
covtr6_3414980
covtr6_3414982!
batch_normalization_2_3414986!
batch_normalization_2_3414988!
batch_normalization_2_3414990!
batch_normalization_2_3414992
covtr7_3414995
covtr7_3414997!
batch_normalization_3_3415001!
batch_normalization_3_3415003!
batch_normalization_3_3415005!
batch_normalization_3_3415007
covtr8_3415010
covtr8_3415012!
batch_normalization_4_3415016!
batch_normalization_4_3415018!
batch_normalization_4_3415020!
batch_normalization_4_3415022
covtr9_3415025
covtr9_3415027!
batch_normalization_5_3415031!
batch_normalization_5_3415033!
batch_normalization_5_3415035!
batch_normalization_5_3415037
covtr10_3415040
covtr10_3415042!
batch_normalization_6_3415046!
batch_normalization_6_3415048!
batch_normalization_6_3415050!
batch_normalization_6_3415052
covtr11_3415055
covtr11_3415057!
batch_normalization_7_3415061!
batch_normalization_7_3415063!
batch_normalization_7_3415065!
batch_normalization_7_3415067
covtr14_3415070
covtr14_3415072
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallвcovtr10/StatefulPartitionedCallвcovtr11/StatefulPartitionedCallвcovtr14/StatefulPartitionedCallвcovtr4/StatefulPartitionedCallвcovtr5/StatefulPartitionedCallвcovtr6/StatefulPartitionedCallвcovtr7/StatefulPartitionedCallвcovtr8/StatefulPartitionedCallвcovtr9/StatefulPartitionedCall▀
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_34141402
reshape/PartitionedCall╞
covtr4/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr4_3414950covtr4_3414952*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr4_layer_call_and_return_conditional_losses_34128952 
covtr4/StatefulPartitionedCallЮ
leaky_re_lu/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_34141582
leaky_re_lu/PartitionedCall╔
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_3414956batch_normalization_3414958batch_normalization_3414960batch_normalization_3414962*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_34129982-
+batch_normalization/StatefulPartitionedCall┌
covtr5/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr5_3414965covtr5_3414967*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr5_layer_call_and_return_conditional_losses_34130472 
covtr5/StatefulPartitionedCallд
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_34142112
leaky_re_lu_1/PartitionedCall┘
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_3414971batch_normalization_1_3414973batch_normalization_1_3414975batch_normalization_1_3414977*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34131502/
-batch_normalization_1/StatefulPartitionedCall▄
covtr6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr6_3414980covtr6_3414982*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr6_layer_call_and_return_conditional_losses_34131992 
covtr6/StatefulPartitionedCallд
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_34142642
leaky_re_lu_2/PartitionedCall┘
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_3414986batch_normalization_2_3414988batch_normalization_2_3414990batch_normalization_2_3414992*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34133022/
-batch_normalization_2/StatefulPartitionedCall▄
covtr7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0covtr7_3414995covtr7_3414997*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr7_layer_call_and_return_conditional_losses_34133512 
covtr7/StatefulPartitionedCallд
leaky_re_lu_3/PartitionedCallPartitionedCall'covtr7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_34143172
leaky_re_lu_3/PartitionedCall┘
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0batch_normalization_3_3415001batch_normalization_3_3415003batch_normalization_3_3415005batch_normalization_3_3415007*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34134542/
-batch_normalization_3/StatefulPartitionedCall▄
covtr8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0covtr8_3415010covtr8_3415012*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr8_layer_call_and_return_conditional_losses_34135032 
covtr8/StatefulPartitionedCallд
leaky_re_lu_4/PartitionedCallPartitionedCall'covtr8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_34143702
leaky_re_lu_4/PartitionedCall┘
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0batch_normalization_4_3415016batch_normalization_4_3415018batch_normalization_4_3415020batch_normalization_4_3415022*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34136062/
-batch_normalization_4/StatefulPartitionedCall▄
covtr9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0covtr9_3415025covtr9_3415027*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr9_layer_call_and_return_conditional_losses_34136552 
covtr9/StatefulPartitionedCallд
leaky_re_lu_5/PartitionedCallPartitionedCall'covtr9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_34144232
leaky_re_lu_5/PartitionedCall┘
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0batch_normalization_5_3415031batch_normalization_5_3415033batch_normalization_5_3415035batch_normalization_5_3415037*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34137582/
-batch_normalization_5/StatefulPartitionedCallс
covtr10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0covtr10_3415040covtr10_3415042*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_covtr10_layer_call_and_return_conditional_losses_34138072!
covtr10/StatefulPartitionedCallе
leaky_re_lu_6/PartitionedCallPartitionedCall(covtr10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_34144762
leaky_re_lu_6/PartitionedCall┘
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0batch_normalization_6_3415046batch_normalization_6_3415048batch_normalization_6_3415050batch_normalization_6_3415052*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34139102/
-batch_normalization_6/StatefulPartitionedCallс
covtr11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0covtr11_3415055covtr11_3415057*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           "*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_covtr11_layer_call_and_return_conditional_losses_34139592!
covtr11/StatefulPartitionedCallе
leaky_re_lu_7/PartitionedCallPartitionedCall(covtr11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           "* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_34145292
leaky_re_lu_7/PartitionedCall┘
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0batch_normalization_7_3415061batch_normalization_7_3415063batch_normalization_7_3415065batch_normalization_7_3415067*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           "*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34140622/
-batch_normalization_7/StatefulPartitionedCallс
covtr14/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0covtr14_3415070covtr14_3415072*
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
GPU2 *0J 8В *M
fHRF
D__inference_covtr14_layer_call_and_return_conditional_losses_34141122!
covtr14/StatefulPartitionedCall└
IdentityIdentity(covtr14/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall ^covtr10/StatefulPartitionedCall ^covtr11/StatefulPartitionedCall ^covtr14/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^covtr5/StatefulPartitionedCall^covtr6/StatefulPartitionedCall^covtr7/StatefulPartitionedCall^covtr8/StatefulPartitionedCall^covtr9/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         с::::::::::::::::::::::::::::::::::::::::::::::::::2Z
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
:         с
 
_user_specified_nameinputs
ч
`
D__inference_reshape_layer_call_and_return_conditional_losses_3414140

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
:         2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*'
_input_shapes
:         с:P L
(
_output_shapes
:         с
 
_user_specified_nameinputs
№╥
└
#__inference__traced_restore_3417068
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
identity_51ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9ё
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*¤
valueєBЁ3B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЇ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesн
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*т
_output_shapes╧
╠:::::::::::::::::::::::::::::::::::::::::::::::::::*A
dtypes7
5232
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЭ
AssignVariableOpAssignVariableOpassignvariableop_covtr4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1г
AssignVariableOp_1AssignVariableOpassignvariableop_1_covtr4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2▒
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3░
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4╖
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5╗
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6е
AssignVariableOp_6AssignVariableOp assignvariableop_6_covtr5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7г
AssignVariableOp_7AssignVariableOpassignvariableop_7_covtr5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8│
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9▓
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10╜
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11┴
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12й
AssignVariableOp_12AssignVariableOp!assignvariableop_12_covtr6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13з
AssignVariableOp_13AssignVariableOpassignvariableop_13_covtr6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14╖
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15╢
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16╜
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17┴
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18й
AssignVariableOp_18AssignVariableOp!assignvariableop_18_covtr7_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19з
AssignVariableOp_19AssignVariableOpassignvariableop_19_covtr7_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20╖
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_3_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21╢
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_3_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22╜
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23┴
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24й
AssignVariableOp_24AssignVariableOp!assignvariableop_24_covtr8_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25з
AssignVariableOp_25AssignVariableOpassignvariableop_25_covtr8_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26╖
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_4_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27╢
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_4_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28╜
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_4_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29┴
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_4_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30й
AssignVariableOp_30AssignVariableOp!assignvariableop_30_covtr9_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31з
AssignVariableOp_31AssignVariableOpassignvariableop_31_covtr9_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32╖
AssignVariableOp_32AssignVariableOp/assignvariableop_32_batch_normalization_5_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33╢
AssignVariableOp_33AssignVariableOp.assignvariableop_33_batch_normalization_5_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34╜
AssignVariableOp_34AssignVariableOp5assignvariableop_34_batch_normalization_5_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35┴
AssignVariableOp_35AssignVariableOp9assignvariableop_35_batch_normalization_5_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36к
AssignVariableOp_36AssignVariableOp"assignvariableop_36_covtr10_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37и
AssignVariableOp_37AssignVariableOp assignvariableop_37_covtr10_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38╖
AssignVariableOp_38AssignVariableOp/assignvariableop_38_batch_normalization_6_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39╢
AssignVariableOp_39AssignVariableOp.assignvariableop_39_batch_normalization_6_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40╜
AssignVariableOp_40AssignVariableOp5assignvariableop_40_batch_normalization_6_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41┴
AssignVariableOp_41AssignVariableOp9assignvariableop_41_batch_normalization_6_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42к
AssignVariableOp_42AssignVariableOp"assignvariableop_42_covtr11_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43и
AssignVariableOp_43AssignVariableOp assignvariableop_43_covtr11_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44╖
AssignVariableOp_44AssignVariableOp/assignvariableop_44_batch_normalization_7_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45╢
AssignVariableOp_45AssignVariableOp.assignvariableop_45_batch_normalization_7_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46╜
AssignVariableOp_46AssignVariableOp5assignvariableop_46_batch_normalization_7_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47┴
AssignVariableOp_47AssignVariableOp9assignvariableop_47_batch_normalization_7_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48к
AssignVariableOp_48AssignVariableOp"assignvariableop_48_covtr14_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49и
AssignVariableOp_49AssignVariableOp assignvariableop_49_covtr14_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_499
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpк	
Identity_50Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_50Э	
Identity_51IdentityIdentity_50:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_51"#
identity_51Identity_51:output:0*▀
_input_shapes═
╩: ::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
М
K
/__inference_leaky_re_lu_1_layer_call_fn_3416227

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
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_34142112
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
°Й
╒
F__inference_Generator_layer_call_and_return_conditional_losses_3414841

inputs
covtr4_3414715
covtr4_3414717
batch_normalization_3414721
batch_normalization_3414723
batch_normalization_3414725
batch_normalization_3414727
covtr5_3414730
covtr5_3414732!
batch_normalization_1_3414736!
batch_normalization_1_3414738!
batch_normalization_1_3414740!
batch_normalization_1_3414742
covtr6_3414745
covtr6_3414747!
batch_normalization_2_3414751!
batch_normalization_2_3414753!
batch_normalization_2_3414755!
batch_normalization_2_3414757
covtr7_3414760
covtr7_3414762!
batch_normalization_3_3414766!
batch_normalization_3_3414768!
batch_normalization_3_3414770!
batch_normalization_3_3414772
covtr8_3414775
covtr8_3414777!
batch_normalization_4_3414781!
batch_normalization_4_3414783!
batch_normalization_4_3414785!
batch_normalization_4_3414787
covtr9_3414790
covtr9_3414792!
batch_normalization_5_3414796!
batch_normalization_5_3414798!
batch_normalization_5_3414800!
batch_normalization_5_3414802
covtr10_3414805
covtr10_3414807!
batch_normalization_6_3414811!
batch_normalization_6_3414813!
batch_normalization_6_3414815!
batch_normalization_6_3414817
covtr11_3414820
covtr11_3414822!
batch_normalization_7_3414826!
batch_normalization_7_3414828!
batch_normalization_7_3414830!
batch_normalization_7_3414832
covtr14_3414835
covtr14_3414837
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallвcovtr10/StatefulPartitionedCallвcovtr11/StatefulPartitionedCallвcovtr14/StatefulPartitionedCallвcovtr4/StatefulPartitionedCallвcovtr5/StatefulPartitionedCallвcovtr6/StatefulPartitionedCallвcovtr7/StatefulPartitionedCallвcovtr8/StatefulPartitionedCallвcovtr9/StatefulPartitionedCall▀
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_34141402
reshape/PartitionedCall╞
covtr4/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr4_3414715covtr4_3414717*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr4_layer_call_and_return_conditional_losses_34128952 
covtr4/StatefulPartitionedCallЮ
leaky_re_lu/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_34141582
leaky_re_lu/PartitionedCall╟
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_3414721batch_normalization_3414723batch_normalization_3414725batch_normalization_3414727*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_34129672-
+batch_normalization/StatefulPartitionedCall┌
covtr5/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr5_3414730covtr5_3414732*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr5_layer_call_and_return_conditional_losses_34130472 
covtr5/StatefulPartitionedCallд
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_34142112
leaky_re_lu_1/PartitionedCall╫
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_3414736batch_normalization_1_3414738batch_normalization_1_3414740batch_normalization_1_3414742*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34131192/
-batch_normalization_1/StatefulPartitionedCall▄
covtr6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr6_3414745covtr6_3414747*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr6_layer_call_and_return_conditional_losses_34131992 
covtr6/StatefulPartitionedCallд
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_34142642
leaky_re_lu_2/PartitionedCall╫
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_3414751batch_normalization_2_3414753batch_normalization_2_3414755batch_normalization_2_3414757*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34132712/
-batch_normalization_2/StatefulPartitionedCall▄
covtr7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0covtr7_3414760covtr7_3414762*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr7_layer_call_and_return_conditional_losses_34133512 
covtr7/StatefulPartitionedCallд
leaky_re_lu_3/PartitionedCallPartitionedCall'covtr7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_34143172
leaky_re_lu_3/PartitionedCall╫
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0batch_normalization_3_3414766batch_normalization_3_3414768batch_normalization_3_3414770batch_normalization_3_3414772*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34134232/
-batch_normalization_3/StatefulPartitionedCall▄
covtr8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0covtr8_3414775covtr8_3414777*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr8_layer_call_and_return_conditional_losses_34135032 
covtr8/StatefulPartitionedCallд
leaky_re_lu_4/PartitionedCallPartitionedCall'covtr8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_34143702
leaky_re_lu_4/PartitionedCall╫
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0batch_normalization_4_3414781batch_normalization_4_3414783batch_normalization_4_3414785batch_normalization_4_3414787*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34135752/
-batch_normalization_4/StatefulPartitionedCall▄
covtr9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0covtr9_3414790covtr9_3414792*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr9_layer_call_and_return_conditional_losses_34136552 
covtr9/StatefulPartitionedCallд
leaky_re_lu_5/PartitionedCallPartitionedCall'covtr9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_34144232
leaky_re_lu_5/PartitionedCall╫
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0batch_normalization_5_3414796batch_normalization_5_3414798batch_normalization_5_3414800batch_normalization_5_3414802*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34137272/
-batch_normalization_5/StatefulPartitionedCallс
covtr10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0covtr10_3414805covtr10_3414807*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_covtr10_layer_call_and_return_conditional_losses_34138072!
covtr10/StatefulPartitionedCallе
leaky_re_lu_6/PartitionedCallPartitionedCall(covtr10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_34144762
leaky_re_lu_6/PartitionedCall╫
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0batch_normalization_6_3414811batch_normalization_6_3414813batch_normalization_6_3414815batch_normalization_6_3414817*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34138792/
-batch_normalization_6/StatefulPartitionedCallс
covtr11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0covtr11_3414820covtr11_3414822*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           "*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_covtr11_layer_call_and_return_conditional_losses_34139592!
covtr11/StatefulPartitionedCallе
leaky_re_lu_7/PartitionedCallPartitionedCall(covtr11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           "* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_34145292
leaky_re_lu_7/PartitionedCall╫
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0batch_normalization_7_3414826batch_normalization_7_3414828batch_normalization_7_3414830batch_normalization_7_3414832*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           "*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34140312/
-batch_normalization_7/StatefulPartitionedCallс
covtr14/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0covtr14_3414835covtr14_3414837*
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
GPU2 *0J 8В *M
fHRF
D__inference_covtr14_layer_call_and_return_conditional_losses_34141122!
covtr14/StatefulPartitionedCall└
IdentityIdentity(covtr14/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall ^covtr10/StatefulPartitionedCall ^covtr11/StatefulPartitionedCall ^covtr14/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^covtr5/StatefulPartitionedCall^covtr6/StatefulPartitionedCall^covtr7/StatefulPartitionedCall^covtr8/StatefulPartitionedCall^covtr9/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         с::::::::::::::::::::::::::::::::::::::::::::::::::2Z
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
:         с
 
_user_specified_nameinputs
├$
╡
C__inference_covtr6_layer_call_and_return_conditional_losses_3413199

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
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpё
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpд
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           :::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╩
п
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3413423

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╩
п
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3416617

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
├$
╡
C__inference_covtr4_layer_call_and_return_conditional_losses_3412895

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
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpё
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpд
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           :::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
к
d
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_3416148

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
И
I
-__inference_leaky_re_lu_layer_call_fn_3416153

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
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_34141582
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
м
f
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_3416222

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╩
п
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_3416543

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
м
f
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_3416666

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           "2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           "2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           ":i e
A
_output_shapes/
-:+                           "
 
_user_specified_nameinputs
д
к
7__inference_batch_normalization_3_layer_call_fn_3416426

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
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34134232
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╩
п
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3413119

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ъ
Л
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3416709

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:"*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:"*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:"*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:"*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           ":":":":":*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           "2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ":::::i e
A
_output_shapes/
-:+                           "
 
_user_specified_nameinputs
Ъ
Л
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3416635

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
М
K
/__inference_leaky_re_lu_6_layer_call_fn_3416597

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
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_34144762
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ъ
Л
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3413910

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╩
п
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3416321

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╩
п
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_3416469

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╔
~
)__inference_covtr14_layer_call_fn_3414122

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
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_covtr14_layer_call_and_return_conditional_losses_34141122
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           "::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           "
 
_user_specified_nameinputs
ч
`
D__inference_reshape_layer_call_and_return_conditional_losses_3416138

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
:         2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*'
_input_shapes
:         с:P L
(
_output_shapes
:         с
 
_user_specified_nameinputs
м
f
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_3414476

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╩
п
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_3413575

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
м
f
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_3416518

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ъ
Л
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_3416487

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╜╦
▄
F__inference_Generator_layer_call_and_return_conditional_losses_3415914

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
identityИT
reshape/ShapeShapeinputs*
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
reshape/Reshape/shape/3ъ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeП
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:         2
reshape/Reshaped
covtr4/ShapeShapereshape/Reshape:output:0*
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
covtr4/strided_slice_1╚
&covtr4/conv2d_transpose/ReadVariableOpReadVariableOp/covtr4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr4/conv2d_transpose/ReadVariableOpН
covtr4/conv2d_transposeConv2DBackpropInputcovtr4/stack:output:0.covtr4/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
covtr4/conv2d_transposeб
covtr4/BiasAdd/ReadVariableOpReadVariableOp&covtr4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr4/BiasAdd/ReadVariableOpо
covtr4/BiasAddBiasAdd covtr4/conv2d_transpose:output:0%covtr4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
covtr4/BiasAddЕ
leaky_re_lu/LeakyRelu	LeakyRelucovtr4/BiasAdd:output:0*/
_output_shapes
:         2
leaky_re_lu/LeakyRelu░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp╢
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1▀
$batch_normalization/FusedBatchNormV3FusedBatchNormV3#leaky_re_lu/LeakyRelu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
is_training( 2&
$batch_normalization/FusedBatchNormV3t
covtr5/ShapeShape(batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr5/ShapeВ
covtr5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr5/strided_slice/stackЖ
covtr5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr5/strided_slice/stack_1Ж
covtr5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr5/strided_slice/stack_2М
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
covtr5/stack/3╝
covtr5/stackPackcovtr5/strided_slice:output:0covtr5/stack/1:output:0covtr5/stack/2:output:0covtr5/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr5/stackЖ
covtr5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr5/strided_slice_1/stackК
covtr5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr5/strided_slice_1/stack_1К
covtr5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr5/strided_slice_1/stack_2Ц
covtr5/strided_slice_1StridedSlicecovtr5/stack:output:0%covtr5/strided_slice_1/stack:output:0'covtr5/strided_slice_1/stack_1:output:0'covtr5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr5/strided_slice_1╚
&covtr5/conv2d_transpose/ReadVariableOpReadVariableOp/covtr5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr5/conv2d_transpose/ReadVariableOpЭ
covtr5/conv2d_transposeConv2DBackpropInputcovtr5/stack:output:0.covtr5/conv2d_transpose/ReadVariableOp:value:0(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
covtr5/conv2d_transposeб
covtr5/BiasAdd/ReadVariableOpReadVariableOp&covtr5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr5/BiasAdd/ReadVariableOpо
covtr5/BiasAddBiasAdd covtr5/conv2d_transpose:output:0%covtr5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
covtr5/BiasAddЙ
leaky_re_lu_1/LeakyRelu	LeakyRelucovtr5/BiasAdd:output:0*/
_output_shapes
:         2
leaky_re_lu_1/LeakyRelu╢
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1э
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_1/LeakyRelu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3v
covtr6/ShapeShape*batch_normalization_1/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr6/ShapeВ
covtr6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr6/strided_slice/stackЖ
covtr6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr6/strided_slice/stack_1Ж
covtr6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr6/strided_slice/stack_2М
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
covtr6/stack/3╝
covtr6/stackPackcovtr6/strided_slice:output:0covtr6/stack/1:output:0covtr6/stack/2:output:0covtr6/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr6/stackЖ
covtr6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr6/strided_slice_1/stackК
covtr6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr6/strided_slice_1/stack_1К
covtr6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr6/strided_slice_1/stack_2Ц
covtr6/strided_slice_1StridedSlicecovtr6/stack:output:0%covtr6/strided_slice_1/stack:output:0'covtr6/strided_slice_1/stack_1:output:0'covtr6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr6/strided_slice_1╚
&covtr6/conv2d_transpose/ReadVariableOpReadVariableOp/covtr6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr6/conv2d_transpose/ReadVariableOpЯ
covtr6/conv2d_transposeConv2DBackpropInputcovtr6/stack:output:0.covtr6/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
covtr6/conv2d_transposeб
covtr6/BiasAdd/ReadVariableOpReadVariableOp&covtr6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr6/BiasAdd/ReadVariableOpо
covtr6/BiasAddBiasAdd covtr6/conv2d_transpose:output:0%covtr6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
covtr6/BiasAddЙ
leaky_re_lu_2/LeakyRelu	LeakyRelucovtr6/BiasAdd:output:0*/
_output_shapes
:         2
leaky_re_lu_2/LeakyRelu╢
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_2/ReadVariableOp╝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1щ
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1э
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_2/LeakyRelu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3v
covtr7/ShapeShape*batch_normalization_2/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr7/ShapeВ
covtr7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr7/strided_slice/stackЖ
covtr7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr7/strided_slice/stack_1Ж
covtr7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr7/strided_slice/stack_2М
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
covtr7/stack/3╝
covtr7/stackPackcovtr7/strided_slice:output:0covtr7/stack/1:output:0covtr7/stack/2:output:0covtr7/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr7/stackЖ
covtr7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr7/strided_slice_1/stackК
covtr7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr7/strided_slice_1/stack_1К
covtr7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr7/strided_slice_1/stack_2Ц
covtr7/strided_slice_1StridedSlicecovtr7/stack:output:0%covtr7/strided_slice_1/stack:output:0'covtr7/strided_slice_1/stack_1:output:0'covtr7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr7/strided_slice_1╚
&covtr7/conv2d_transpose/ReadVariableOpReadVariableOp/covtr7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr7/conv2d_transpose/ReadVariableOpЯ
covtr7/conv2d_transposeConv2DBackpropInputcovtr7/stack:output:0.covtr7/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
covtr7/conv2d_transposeб
covtr7/BiasAdd/ReadVariableOpReadVariableOp&covtr7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr7/BiasAdd/ReadVariableOpо
covtr7/BiasAddBiasAdd covtr7/conv2d_transpose:output:0%covtr7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
covtr7/BiasAddЙ
leaky_re_lu_3/LeakyRelu	LeakyRelucovtr7/BiasAdd:output:0*/
_output_shapes
:         2
leaky_re_lu_3/LeakyRelu╢
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_3/ReadVariableOp╝
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_3/ReadVariableOp_1щ
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1э
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_3/LeakyRelu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3v
covtr8/ShapeShape*batch_normalization_3/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr8/ShapeВ
covtr8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr8/strided_slice/stackЖ
covtr8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr8/strided_slice/stack_1Ж
covtr8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr8/strided_slice/stack_2М
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
covtr8/stack/3╝
covtr8/stackPackcovtr8/strided_slice:output:0covtr8/stack/1:output:0covtr8/stack/2:output:0covtr8/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr8/stackЖ
covtr8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr8/strided_slice_1/stackК
covtr8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr8/strided_slice_1/stack_1К
covtr8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr8/strided_slice_1/stack_2Ц
covtr8/strided_slice_1StridedSlicecovtr8/stack:output:0%covtr8/strided_slice_1/stack:output:0'covtr8/strided_slice_1/stack_1:output:0'covtr8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr8/strided_slice_1╚
&covtr8/conv2d_transpose/ReadVariableOpReadVariableOp/covtr8_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr8/conv2d_transpose/ReadVariableOpЯ
covtr8/conv2d_transposeConv2DBackpropInputcovtr8/stack:output:0.covtr8/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
covtr8/conv2d_transposeб
covtr8/BiasAdd/ReadVariableOpReadVariableOp&covtr8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr8/BiasAdd/ReadVariableOpо
covtr8/BiasAddBiasAdd covtr8/conv2d_transpose:output:0%covtr8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
covtr8/BiasAddЙ
leaky_re_lu_4/LeakyRelu	LeakyRelucovtr8/BiasAdd:output:0*/
_output_shapes
:         2
leaky_re_lu_4/LeakyRelu╢
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_4/ReadVariableOp╝
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1щ
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1э
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_4/LeakyRelu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3v
covtr9/ShapeShape*batch_normalization_4/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr9/ShapeВ
covtr9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr9/strided_slice/stackЖ
covtr9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr9/strided_slice/stack_1Ж
covtr9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr9/strided_slice/stack_2М
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
covtr9/stack/3╝
covtr9/stackPackcovtr9/strided_slice:output:0covtr9/stack/1:output:0covtr9/stack/2:output:0covtr9/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr9/stackЖ
covtr9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr9/strided_slice_1/stackК
covtr9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr9/strided_slice_1/stack_1К
covtr9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
covtr9/strided_slice_1/stack_2Ц
covtr9/strided_slice_1StridedSlicecovtr9/stack:output:0%covtr9/strided_slice_1/stack:output:0'covtr9/strided_slice_1/stack_1:output:0'covtr9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr9/strided_slice_1╚
&covtr9/conv2d_transpose/ReadVariableOpReadVariableOp/covtr9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02(
&covtr9/conv2d_transpose/ReadVariableOpЯ
covtr9/conv2d_transposeConv2DBackpropInputcovtr9/stack:output:0.covtr9/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         $*
paddingVALID*
strides
2
covtr9/conv2d_transposeб
covtr9/BiasAdd/ReadVariableOpReadVariableOp&covtr9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
covtr9/BiasAdd/ReadVariableOpо
covtr9/BiasAddBiasAdd covtr9/conv2d_transpose:output:0%covtr9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         $2
covtr9/BiasAddЙ
leaky_re_lu_5/LeakyRelu	LeakyRelucovtr9/BiasAdd:output:0*/
_output_shapes
:         $2
leaky_re_lu_5/LeakyRelu╢
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_5/ReadVariableOp╝
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_5/ReadVariableOp_1щ
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1э
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_5/LeakyRelu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         $:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3x
covtr10/ShapeShape*batch_normalization_5/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr10/ShapeД
covtr10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr10/strided_slice/stackИ
covtr10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr10/strided_slice/stack_1И
covtr10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr10/strided_slice/stack_2Т
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
covtr10/stack/3┬
covtr10/stackPackcovtr10/strided_slice:output:0covtr10/stack/1:output:0covtr10/stack/2:output:0covtr10/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr10/stackИ
covtr10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr10/strided_slice_1/stackМ
covtr10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr10/strided_slice_1/stack_1М
covtr10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr10/strided_slice_1/stack_2Ь
covtr10/strided_slice_1StridedSlicecovtr10/stack:output:0&covtr10/strided_slice_1/stack:output:0(covtr10/strided_slice_1/stack_1:output:0(covtr10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr10/strided_slice_1╦
'covtr10/conv2d_transpose/ReadVariableOpReadVariableOp0covtr10_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02)
'covtr10/conv2d_transpose/ReadVariableOpг
covtr10/conv2d_transposeConv2DBackpropInputcovtr10/stack:output:0/covtr10/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         +!*
paddingVALID*
strides
2
covtr10/conv2d_transposeд
covtr10/BiasAdd/ReadVariableOpReadVariableOp'covtr10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
covtr10/BiasAdd/ReadVariableOp▓
covtr10/BiasAddBiasAdd!covtr10/conv2d_transpose:output:0&covtr10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         +!2
covtr10/BiasAddК
leaky_re_lu_6/LeakyRelu	LeakyRelucovtr10/BiasAdd:output:0*/
_output_shapes
:         +!2
leaky_re_lu_6/LeakyRelu╢
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_6/ReadVariableOp╝
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1щ
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1э
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_6/LeakyRelu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         +!:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3x
covtr11/ShapeShape*batch_normalization_6/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr11/ShapeД
covtr11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr11/strided_slice/stackИ
covtr11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr11/strided_slice/stack_1И
covtr11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr11/strided_slice/stack_2Т
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
covtr11/stack/3┬
covtr11/stackPackcovtr11/strided_slice:output:0covtr11/stack/1:output:0covtr11/stack/2:output:0covtr11/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr11/stackИ
covtr11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr11/strided_slice_1/stackМ
covtr11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr11/strided_slice_1/stack_1М
covtr11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr11/strided_slice_1/stack_2Ь
covtr11/strided_slice_1StridedSlicecovtr11/stack:output:0&covtr11/strided_slice_1/stack:output:0(covtr11/strided_slice_1/stack_1:output:0(covtr11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr11/strided_slice_1╦
'covtr11/conv2d_transpose/ReadVariableOpReadVariableOp0covtr11_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		"*
dtype02)
'covtr11/conv2d_transpose/ReadVariableOpг
covtr11/conv2d_transposeConv2DBackpropInputcovtr11/stack:output:0/covtr11/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         3)"*
paddingVALID*
strides
2
covtr11/conv2d_transposeд
covtr11/BiasAdd/ReadVariableOpReadVariableOp'covtr11_biasadd_readvariableop_resource*
_output_shapes
:"*
dtype02 
covtr11/BiasAdd/ReadVariableOp▓
covtr11/BiasAddBiasAdd!covtr11/conv2d_transpose:output:0&covtr11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         3)"2
covtr11/BiasAddК
leaky_re_lu_7/LeakyRelu	LeakyRelucovtr11/BiasAdd:output:0*/
_output_shapes
:         3)"2
leaky_re_lu_7/LeakyRelu╢
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:"*
dtype02&
$batch_normalization_7/ReadVariableOp╝
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:"*
dtype02(
&batch_normalization_7/ReadVariableOp_1щ
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:"*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:"*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1э
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_7/LeakyRelu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         3)":":":":":*
epsilon%oГ:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3x
covtr14/ShapeShape*batch_normalization_7/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
covtr14/ShapeД
covtr14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr14/strided_slice/stackИ
covtr14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
covtr14/strided_slice/stack_1И
covtr14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
covtr14/strided_slice/stack_2Т
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
covtr14/stack/3┬
covtr14/stackPackcovtr14/strided_slice:output:0covtr14/stack/1:output:0covtr14/stack/2:output:0covtr14/stack/3:output:0*
N*
T0*
_output_shapes
:2
covtr14/stackИ
covtr14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
covtr14/strided_slice_1/stackМ
covtr14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr14/strided_slice_1/stack_1М
covtr14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
covtr14/strided_slice_1/stack_2Ь
covtr14/strided_slice_1StridedSlicecovtr14/stack:output:0&covtr14/strided_slice_1/stack:output:0(covtr14/strided_slice_1/stack_1:output:0(covtr14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
covtr14/strided_slice_1╦
'covtr14/conv2d_transpose/ReadVariableOpReadVariableOp0covtr14_conv2d_transpose_readvariableop_resource*&
_output_shapes
:"*
dtype02)
'covtr14/conv2d_transpose/ReadVariableOpг
covtr14/conv2d_transposeConv2DBackpropInputcovtr14/stack:output:0/covtr14/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         7-*
paddingVALID*
strides
2
covtr14/conv2d_transposeд
covtr14/BiasAdd/ReadVariableOpReadVariableOp'covtr14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
covtr14/BiasAdd/ReadVariableOp▓
covtr14/BiasAddBiasAdd!covtr14/conv2d_transpose:output:0&covtr14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         7-2
covtr14/BiasAddБ
covtr14/SigmoidSigmoidcovtr14/BiasAdd:output:0*
T0*/
_output_shapes
:         7-2
covtr14/Sigmoido
IdentityIdentitycovtr14/Sigmoid:y:0*
T0*/
_output_shapes
:         7-2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         с:::::::::::::::::::::::::::::::::::::::::::::::::::P L
(
_output_shapes
:         с
 
_user_specified_nameinputs
╩
п
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_3413727

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ъ
Л
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3414062

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:"*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:"*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:"*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:"*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           ":":":":":*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           "2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ":::::i e
A
_output_shapes/
-:+                           "
 
_user_specified_nameinputs
ж
к
7__inference_batch_normalization_7_layer_call_fn_3416735

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
-:+                           "*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34140622
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           "2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           "::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           "
 
_user_specified_nameinputs
├$
╡
C__inference_covtr5_layer_call_and_return_conditional_losses_3413047

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
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpё
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpд
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           :::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
 f
Н
 __inference__traced_save_3416908
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
value3B1 B+_temp_d42949bcf4d44e8289fab73ae6efbd15/part2	
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
ShardedFilenameы
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*¤
valueєBЁ3B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesю
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices─
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_covtr4_kernel_read_readvariableop&savev2_covtr4_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop(savev2_covtr5_kernel_read_readvariableop&savev2_covtr5_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop(savev2_covtr6_kernel_read_readvariableop&savev2_covtr6_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop(savev2_covtr7_kernel_read_readvariableop&savev2_covtr7_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop(savev2_covtr8_kernel_read_readvariableop&savev2_covtr8_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop(savev2_covtr9_kernel_read_readvariableop&savev2_covtr9_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop)savev2_covtr10_kernel_read_readvariableop'savev2_covtr10_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop)savev2_covtr11_kernel_read_readvariableop'savev2_covtr11_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop)savev2_covtr14_kernel_read_readvariableop'savev2_covtr14_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *A
dtypes7
5232
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

identity_1Identity_1:output:0*▒
_input_shapesЯ
Ь: :::::::::::::::::::::::::::::::::::::::::::		":":":":":":":: 2(
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
м
f
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_3414317

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
М
K
/__inference_leaky_re_lu_4_layer_call_fn_3416449

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
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_34143702
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ш
Й
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3416191

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ш
Й
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3412998

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
├$
╡
C__inference_covtr9_layer_call_and_return_conditional_losses_3413655

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
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpё
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpд
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           :::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
д
к
7__inference_batch_normalization_7_layer_call_fn_3416722

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
-:+                           "*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34140312
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           "2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           "::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           "
 
_user_specified_nameinputs
м
f
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_3414529

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           "2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           "2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           ":i e
A
_output_shapes/
-:+                           "
 
_user_specified_nameinputs
М
K
/__inference_leaky_re_lu_3_layer_call_fn_3416375

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
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_34143172
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╩
п
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3416247

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╚
н
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3416173

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
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
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
─$
╢
D__inference_covtr10_layer_call_and_return_conditional_losses_3413807

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
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpё
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpд
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           :::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ж
к
7__inference_batch_normalization_1_layer_call_fn_3416291

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
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34131502
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
СК
╪
F__inference_Generator_layer_call_and_return_conditional_losses_3414708
	gen_noise
covtr4_3414582
covtr4_3414584
batch_normalization_3414588
batch_normalization_3414590
batch_normalization_3414592
batch_normalization_3414594
covtr5_3414597
covtr5_3414599!
batch_normalization_1_3414603!
batch_normalization_1_3414605!
batch_normalization_1_3414607!
batch_normalization_1_3414609
covtr6_3414612
covtr6_3414614!
batch_normalization_2_3414618!
batch_normalization_2_3414620!
batch_normalization_2_3414622!
batch_normalization_2_3414624
covtr7_3414627
covtr7_3414629!
batch_normalization_3_3414633!
batch_normalization_3_3414635!
batch_normalization_3_3414637!
batch_normalization_3_3414639
covtr8_3414642
covtr8_3414644!
batch_normalization_4_3414648!
batch_normalization_4_3414650!
batch_normalization_4_3414652!
batch_normalization_4_3414654
covtr9_3414657
covtr9_3414659!
batch_normalization_5_3414663!
batch_normalization_5_3414665!
batch_normalization_5_3414667!
batch_normalization_5_3414669
covtr10_3414672
covtr10_3414674!
batch_normalization_6_3414678!
batch_normalization_6_3414680!
batch_normalization_6_3414682!
batch_normalization_6_3414684
covtr11_3414687
covtr11_3414689!
batch_normalization_7_3414693!
batch_normalization_7_3414695!
batch_normalization_7_3414697!
batch_normalization_7_3414699
covtr14_3414702
covtr14_3414704
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallвcovtr10/StatefulPartitionedCallвcovtr11/StatefulPartitionedCallвcovtr14/StatefulPartitionedCallвcovtr4/StatefulPartitionedCallвcovtr5/StatefulPartitionedCallвcovtr6/StatefulPartitionedCallвcovtr7/StatefulPartitionedCallвcovtr8/StatefulPartitionedCallвcovtr9/StatefulPartitionedCallт
reshape/PartitionedCallPartitionedCall	gen_noise*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_34141402
reshape/PartitionedCall╞
covtr4/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0covtr4_3414582covtr4_3414584*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr4_layer_call_and_return_conditional_losses_34128952 
covtr4/StatefulPartitionedCallЮ
leaky_re_lu/PartitionedCallPartitionedCall'covtr4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_34141582
leaky_re_lu/PartitionedCall╔
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_3414588batch_normalization_3414590batch_normalization_3414592batch_normalization_3414594*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_34129982-
+batch_normalization/StatefulPartitionedCall┌
covtr5/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0covtr5_3414597covtr5_3414599*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr5_layer_call_and_return_conditional_losses_34130472 
covtr5/StatefulPartitionedCallд
leaky_re_lu_1/PartitionedCallPartitionedCall'covtr5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_34142112
leaky_re_lu_1/PartitionedCall┘
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_3414603batch_normalization_1_3414605batch_normalization_1_3414607batch_normalization_1_3414609*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34131502/
-batch_normalization_1/StatefulPartitionedCall▄
covtr6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0covtr6_3414612covtr6_3414614*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr6_layer_call_and_return_conditional_losses_34131992 
covtr6/StatefulPartitionedCallд
leaky_re_lu_2/PartitionedCallPartitionedCall'covtr6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_34142642
leaky_re_lu_2/PartitionedCall┘
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_3414618batch_normalization_2_3414620batch_normalization_2_3414622batch_normalization_2_3414624*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34133022/
-batch_normalization_2/StatefulPartitionedCall▄
covtr7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0covtr7_3414627covtr7_3414629*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr7_layer_call_and_return_conditional_losses_34133512 
covtr7/StatefulPartitionedCallд
leaky_re_lu_3/PartitionedCallPartitionedCall'covtr7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_34143172
leaky_re_lu_3/PartitionedCall┘
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0batch_normalization_3_3414633batch_normalization_3_3414635batch_normalization_3_3414637batch_normalization_3_3414639*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34134542/
-batch_normalization_3/StatefulPartitionedCall▄
covtr8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0covtr8_3414642covtr8_3414644*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr8_layer_call_and_return_conditional_losses_34135032 
covtr8/StatefulPartitionedCallд
leaky_re_lu_4/PartitionedCallPartitionedCall'covtr8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_34143702
leaky_re_lu_4/PartitionedCall┘
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0batch_normalization_4_3414648batch_normalization_4_3414650batch_normalization_4_3414652batch_normalization_4_3414654*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34136062/
-batch_normalization_4/StatefulPartitionedCall▄
covtr9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0covtr9_3414657covtr9_3414659*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr9_layer_call_and_return_conditional_losses_34136552 
covtr9/StatefulPartitionedCallд
leaky_re_lu_5/PartitionedCallPartitionedCall'covtr9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_34144232
leaky_re_lu_5/PartitionedCall┘
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0batch_normalization_5_3414663batch_normalization_5_3414665batch_normalization_5_3414667batch_normalization_5_3414669*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34137582/
-batch_normalization_5/StatefulPartitionedCallс
covtr10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0covtr10_3414672covtr10_3414674*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_covtr10_layer_call_and_return_conditional_losses_34138072!
covtr10/StatefulPartitionedCallе
leaky_re_lu_6/PartitionedCallPartitionedCall(covtr10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_34144762
leaky_re_lu_6/PartitionedCall┘
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0batch_normalization_6_3414678batch_normalization_6_3414680batch_normalization_6_3414682batch_normalization_6_3414684*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34139102/
-batch_normalization_6/StatefulPartitionedCallс
covtr11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0covtr11_3414687covtr11_3414689*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           "*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_covtr11_layer_call_and_return_conditional_losses_34139592!
covtr11/StatefulPartitionedCallе
leaky_re_lu_7/PartitionedCallPartitionedCall(covtr11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           "* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_34145292
leaky_re_lu_7/PartitionedCall┘
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0batch_normalization_7_3414693batch_normalization_7_3414695batch_normalization_7_3414697batch_normalization_7_3414699*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           "*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34140622/
-batch_normalization_7/StatefulPartitionedCallс
covtr14/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0covtr14_3414702covtr14_3414704*
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
GPU2 *0J 8В *M
fHRF
D__inference_covtr14_layer_call_and_return_conditional_losses_34141122!
covtr14/StatefulPartitionedCall└
IdentityIdentity(covtr14/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall ^covtr10/StatefulPartitionedCall ^covtr11/StatefulPartitionedCall ^covtr14/StatefulPartitionedCall^covtr4/StatefulPartitionedCall^covtr5/StatefulPartitionedCall^covtr6/StatefulPartitionedCall^covtr7/StatefulPartitionedCall^covtr8/StatefulPartitionedCall^covtr9/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         с::::::::::::::::::::::::::::::::::::::::::::::::::2Z
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
:         с
#
_user_specified_name	gen_noise
Ъ
Л
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3413150

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
д
к
7__inference_batch_normalization_6_layer_call_fn_3416648

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
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34138792
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╔
~
)__inference_covtr10_layer_call_fn_3413817

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
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_covtr10_layer_call_and_return_conditional_losses_34138072
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╟
}
(__inference_covtr9_layer_call_fn_3413665

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
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr9_layer_call_and_return_conditional_losses_34136552
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
м
f
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_3414423

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
м
f
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_3414211

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╟
}
(__inference_covtr5_layer_call_fn_3413057

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
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_covtr5_layer_call_and_return_conditional_losses_34130472
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs"╕L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╖
serving_defaultг
@
	gen_noise3
serving_default_gen_noise:0         сC
covtr148
StatefulPartitionedCall:0         7-tensorflow/serving/predict:╠╫
╜я
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
trainable_variables
regularization_losses
	keras_api
 
signatures
╩__call__
+╦&call_and_return_all_conditional_losses
╠_default_save_signature"╠ч
_tf_keras_networkпч{"class_name": "Functional", "name": "Generator", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Generator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 225]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gen_noise"}, "name": "gen_noise", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [15, 5, 3]}}, "name": "reshape", "inbound_nodes": [[["gen_noise", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr4", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr4", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu", "inbound_nodes": [[["covtr4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr5", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr5", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_1", "inbound_nodes": [[["covtr5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr6", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr6", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_2", "inbound_nodes": [[["covtr6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr7", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr7", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_3", "inbound_nodes": [[["covtr7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["leaky_re_lu_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr8", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr8", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_4", "inbound_nodes": [[["covtr8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr9", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr9", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_5", "inbound_nodes": [[["covtr9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["leaky_re_lu_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr10", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr10", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_6", "inbound_nodes": [[["covtr10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["leaky_re_lu_6", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr11", "trainable": true, "dtype": "float32", "filters": 34, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr11", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_7", "inbound_nodes": [[["covtr11", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["leaky_re_lu_7", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr14", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr14", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}], "input_layers": [["gen_noise", 0, 0]], "output_layers": [["covtr14", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 225]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Generator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 225]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gen_noise"}, "name": "gen_noise", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [15, 5, 3]}}, "name": "reshape", "inbound_nodes": [[["gen_noise", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr4", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr4", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu", "inbound_nodes": [[["covtr4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr5", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr5", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_1", "inbound_nodes": [[["covtr5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr6", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr6", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_2", "inbound_nodes": [[["covtr6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr7", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr7", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_3", "inbound_nodes": [[["covtr7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["leaky_re_lu_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr8", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr8", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_4", "inbound_nodes": [[["covtr8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr9", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr9", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_5", "inbound_nodes": [[["covtr9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["leaky_re_lu_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr10", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr10", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_6", "inbound_nodes": [[["covtr10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["leaky_re_lu_6", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr11", "trainable": true, "dtype": "float32", "filters": 34, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr11", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_7", "inbound_nodes": [[["covtr11", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["leaky_re_lu_7", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "covtr14", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "covtr14", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}], "input_layers": [["gen_noise", 0, 0]], "output_layers": [["covtr14", 0, 0]]}}}
ё"ю
_tf_keras_input_layer╬{"class_name": "InputLayer", "name": "gen_noise", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 225]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 225]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gen_noise"}}
Ў
!	variables
"trainable_variables
#regularization_losses
$	keras_api
═__call__
+╬&call_and_return_all_conditional_losses"х
_tf_keras_layer╦{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [15, 5, 3]}}}
м


%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
╧__call__
+╨&call_and_return_all_conditional_losses"Е	
_tf_keras_layerы{"class_name": "Conv2DTranspose", "name": "covtr4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr4", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 5, 3]}}
▄
+	variables
,trainable_variables
-regularization_losses
.	keras_api
╤__call__
+╥&call_and_return_all_conditional_losses"╦
_tf_keras_layer▒{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
╡	
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4	variables
5trainable_variables
6regularization_losses
7	keras_api
╙__call__
+╘&call_and_return_all_conditional_losses"▀
_tf_keras_layer┼{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 6, 4]}}
м


8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
╒__call__
+╓&call_and_return_all_conditional_losses"Е	
_tf_keras_layerы{"class_name": "Conv2DTranspose", "name": "covtr5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr5", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 6, 4]}}
р
>	variables
?trainable_variables
@regularization_losses
A	keras_api
╫__call__
+╪&call_and_return_all_conditional_losses"╧
_tf_keras_layer╡{"class_name": "LeakyReLU", "name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
╣	
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
┘__call__
+┌&call_and_return_all_conditional_losses"у
_tf_keras_layer╔{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18, 8, 6]}}
м


Kkernel
Lbias
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
█__call__
+▄&call_and_return_all_conditional_losses"Е	
_tf_keras_layerы{"class_name": "Conv2DTranspose", "name": "covtr6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr6", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18, 8, 6]}}
р
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
▌__call__
+▐&call_and_return_all_conditional_losses"╧
_tf_keras_layer╡{"class_name": "LeakyReLU", "name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
║	
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
▀__call__
+р&call_and_return_all_conditional_losses"ф
_tf_keras_layer╩{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21, 11, 8]}}
о


^kernel
_bias
`	variables
atrainable_variables
bregularization_losses
c	keras_api
с__call__
+т&call_and_return_all_conditional_losses"З	
_tf_keras_layerэ{"class_name": "Conv2DTranspose", "name": "covtr7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr7", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21, 11, 8]}}
р
d	variables
etrainable_variables
fregularization_losses
g	keras_api
у__call__
+ф&call_and_return_all_conditional_losses"╧
_tf_keras_layer╡{"class_name": "LeakyReLU", "name": "leaky_re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
╝	
haxis
	igamma
jbeta
kmoving_mean
lmoving_variance
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
х__call__
+ц&call_and_return_all_conditional_losses"ц
_tf_keras_layer╠{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25, 15, 16]}}
░


qkernel
rbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses"Й	
_tf_keras_layerя{"class_name": "Conv2DTranspose", "name": "covtr8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr8", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25, 15, 16]}}
р
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses"╧
_tf_keras_layer╡{"class_name": "LeakyReLU", "name": "leaky_re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
└	
{axis
	|gamma
}beta
~moving_mean
moving_variance
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses"ц
_tf_keras_layer╠{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 20, 20]}}
╢

Дkernel
	Еbias
Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api
э__call__
+ю&call_and_return_all_conditional_losses"Й	
_tf_keras_layerя{"class_name": "Conv2DTranspose", "name": "covtr9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr9", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 20, 20]}}
ф
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
я__call__
+Ё&call_and_return_all_conditional_losses"╧
_tf_keras_layer╡{"class_name": "LeakyReLU", "name": "leaky_re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
┼	
	Оaxis

Пgamma
	Рbeta
Сmoving_mean
Тmoving_variance
У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
ё__call__
+Є&call_and_return_all_conditional_losses"ц
_tf_keras_layer╠{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36, 26, 24]}}
╕

Чkernel
	Шbias
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
є__call__
+Ї&call_and_return_all_conditional_losses"Л	
_tf_keras_layerё{"class_name": "Conv2DTranspose", "name": "covtr10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr10", "trainable": true, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36, 26, 24]}}
ф
Э	variables
Юtrainable_variables
Яregularization_losses
а	keras_api
ї__call__
+Ў&call_and_return_all_conditional_losses"╧
_tf_keras_layer╡{"class_name": "LeakyReLU", "name": "leaky_re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
┼	
	бaxis

вgamma
	гbeta
дmoving_mean
еmoving_variance
ж	variables
зtrainable_variables
иregularization_losses
й	keras_api
ў__call__
+°&call_and_return_all_conditional_losses"ц
_tf_keras_layer╠{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 43, 33, 30]}}
╕

кkernel
	лbias
м	variables
нtrainable_variables
оregularization_losses
п	keras_api
∙__call__
+·&call_and_return_all_conditional_losses"Л	
_tf_keras_layerё{"class_name": "Conv2DTranspose", "name": "covtr11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr11", "trainable": true, "dtype": "float32", "filters": 34, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 43, 33, 30]}}
ф
░	variables
▒trainable_variables
▓regularization_losses
│	keras_api
√__call__
+№&call_and_return_all_conditional_losses"╧
_tf_keras_layer╡{"class_name": "LeakyReLU", "name": "leaky_re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
┼	
	┤axis

╡gamma
	╢beta
╖moving_mean
╕moving_variance
╣	variables
║trainable_variables
╗regularization_losses
╝	keras_api
¤__call__
+■&call_and_return_all_conditional_losses"ц
_tf_keras_layer╠{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 34}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 51, 41, 34]}}
╕

╜kernel
	╛bias
┐	variables
└trainable_variables
┴regularization_losses
┬	keras_api
 __call__
+А&call_and_return_all_conditional_losses"Л	
_tf_keras_layerё{"class_name": "Conv2DTranspose", "name": "covtr14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "covtr14", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 34}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 51, 41, 34]}}
║
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
Д30
Е31
П32
Р33
С34
Т35
Ч36
Ш37
в38
г39
д40
е41
к42
л43
╡44
╢45
╖46
╕47
╜48
╛49"
trackable_list_wrapper
┤
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
Д20
Е21
П22
Р23
Ч24
Ш25
в26
г27
к28
л29
╡30
╢31
╜32
╛33"
trackable_list_wrapper
 "
trackable_list_wrapper
╙
├layer_metrics
─non_trainable_variables
	variables
 ┼layer_regularization_losses
╞layers
trainable_variables
╟metrics
regularization_losses
╩__call__
╠_default_save_signature
+╦&call_and_return_all_conditional_losses
'╦"call_and_return_conditional_losses"
_generic_user_object
-
Бserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╚layer_metrics
╔non_trainable_variables
 ╩layer_regularization_losses
!	variables
╦layers
"trainable_variables
╠metrics
#regularization_losses
═__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses"
_generic_user_object
':%2covtr4/kernel
:2covtr4/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
═layer_metrics
╬non_trainable_variables
 ╧layer_regularization_losses
'	variables
╨layers
(trainable_variables
╤metrics
)regularization_losses
╧__call__
+╨&call_and_return_all_conditional_losses
'╨"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╥layer_metrics
╙non_trainable_variables
 ╘layer_regularization_losses
+	variables
╒layers
,trainable_variables
╓metrics
-regularization_losses
╤__call__
+╥&call_and_return_all_conditional_losses
'╥"call_and_return_conditional_losses"
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
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╫layer_metrics
╪non_trainable_variables
 ┘layer_regularization_losses
4	variables
┌layers
5trainable_variables
█metrics
6regularization_losses
╙__call__
+╘&call_and_return_all_conditional_losses
'╘"call_and_return_conditional_losses"
_generic_user_object
':%2covtr5/kernel
:2covtr5/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
▄layer_metrics
▌non_trainable_variables
 ▐layer_regularization_losses
:	variables
▀layers
;trainable_variables
рmetrics
<regularization_losses
╒__call__
+╓&call_and_return_all_conditional_losses
'╓"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
сlayer_metrics
тnon_trainable_variables
 уlayer_regularization_losses
>	variables
фlayers
?trainable_variables
хmetrics
@regularization_losses
╫__call__
+╪&call_and_return_all_conditional_losses
'╪"call_and_return_conditional_losses"
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
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
цlayer_metrics
чnon_trainable_variables
 шlayer_regularization_losses
G	variables
щlayers
Htrainable_variables
ъmetrics
Iregularization_losses
┘__call__
+┌&call_and_return_all_conditional_losses
'┌"call_and_return_conditional_losses"
_generic_user_object
':%2covtr6/kernel
:2covtr6/bias
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
ыlayer_metrics
ьnon_trainable_variables
 эlayer_regularization_losses
M	variables
юlayers
Ntrainable_variables
яmetrics
Oregularization_losses
█__call__
+▄&call_and_return_all_conditional_losses
'▄"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Ёlayer_metrics
ёnon_trainable_variables
 Єlayer_regularization_losses
Q	variables
єlayers
Rtrainable_variables
Їmetrics
Sregularization_losses
▌__call__
+▐&call_and_return_all_conditional_losses
'▐"call_and_return_conditional_losses"
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
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
їlayer_metrics
Ўnon_trainable_variables
 ўlayer_regularization_losses
Z	variables
°layers
[trainable_variables
∙metrics
\regularization_losses
▀__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
':%2covtr7/kernel
:2covtr7/bias
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
·layer_metrics
√non_trainable_variables
 №layer_regularization_losses
`	variables
¤layers
atrainable_variables
■metrics
bregularization_losses
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 layer_metrics
Аnon_trainable_variables
 Бlayer_regularization_losses
d	variables
Вlayers
etrainable_variables
Гmetrics
fregularization_losses
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
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
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Дlayer_metrics
Еnon_trainable_variables
 Жlayer_regularization_losses
m	variables
Зlayers
ntrainable_variables
Иmetrics
oregularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
':%2covtr8/kernel
:2covtr8/bias
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Йlayer_metrics
Кnon_trainable_variables
 Лlayer_regularization_losses
s	variables
Мlayers
ttrainable_variables
Нmetrics
uregularization_losses
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Оlayer_metrics
Пnon_trainable_variables
 Рlayer_regularization_losses
w	variables
Сlayers
xtrainable_variables
Тmetrics
yregularization_losses
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
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
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Уlayer_metrics
Фnon_trainable_variables
 Хlayer_regularization_losses
А	variables
Цlayers
Бtrainable_variables
Чmetrics
Вregularization_losses
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
':%2covtr9/kernel
:2covtr9/bias
0
Д0
Е1"
trackable_list_wrapper
0
Д0
Е1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Шlayer_metrics
Щnon_trainable_variables
 Ъlayer_regularization_losses
Ж	variables
Ыlayers
Зtrainable_variables
Ьmetrics
Иregularization_losses
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Эlayer_metrics
Юnon_trainable_variables
 Яlayer_regularization_losses
К	variables
аlayers
Лtrainable_variables
бmetrics
Мregularization_losses
я__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_5/gamma
(:&2batch_normalization_5/beta
1:/ (2!batch_normalization_5/moving_mean
5:3 (2%batch_normalization_5/moving_variance
@
П0
Р1
С2
Т3"
trackable_list_wrapper
0
П0
Р1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
вlayer_metrics
гnon_trainable_variables
 дlayer_regularization_losses
У	variables
еlayers
Фtrainable_variables
жmetrics
Хregularization_losses
ё__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
(:&2covtr10/kernel
:2covtr10/bias
0
Ч0
Ш1"
trackable_list_wrapper
0
Ч0
Ш1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
зlayer_metrics
иnon_trainable_variables
 йlayer_regularization_losses
Щ	variables
кlayers
Ъtrainable_variables
лmetrics
Ыregularization_losses
є__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
мlayer_metrics
нnon_trainable_variables
 оlayer_regularization_losses
Э	variables
пlayers
Юtrainable_variables
░metrics
Яregularization_losses
ї__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_6/gamma
(:&2batch_normalization_6/beta
1:/ (2!batch_normalization_6/moving_mean
5:3 (2%batch_normalization_6/moving_variance
@
в0
г1
д2
е3"
trackable_list_wrapper
0
в0
г1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
▒layer_metrics
▓non_trainable_variables
 │layer_regularization_losses
ж	variables
┤layers
зtrainable_variables
╡metrics
иregularization_losses
ў__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
(:&		"2covtr11/kernel
:"2covtr11/bias
0
к0
л1"
trackable_list_wrapper
0
к0
л1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╢layer_metrics
╖non_trainable_variables
 ╕layer_regularization_losses
м	variables
╣layers
нtrainable_variables
║metrics
оregularization_losses
∙__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╗layer_metrics
╝non_trainable_variables
 ╜layer_regularization_losses
░	variables
╛layers
▒trainable_variables
┐metrics
▓regularization_losses
√__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'"2batch_normalization_7/gamma
(:&"2batch_normalization_7/beta
1:/" (2!batch_normalization_7/moving_mean
5:3" (2%batch_normalization_7/moving_variance
@
╡0
╢1
╖2
╕3"
trackable_list_wrapper
0
╡0
╢1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
└layer_metrics
┴non_trainable_variables
 ┬layer_regularization_losses
╣	variables
├layers
║trainable_variables
─metrics
╗regularization_losses
¤__call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses"
_generic_user_object
(:&"2covtr14/kernel
:2covtr14/bias
0
╜0
╛1"
trackable_list_wrapper
0
╜0
╛1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┼layer_metrics
╞non_trainable_variables
 ╟layer_regularization_losses
┐	variables
╚layers
└trainable_variables
╔metrics
┴regularization_losses
 __call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
Ь
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
С10
Т11
д12
е13
╖14
╕15"
trackable_list_wrapper
 "
trackable_list_wrapper
ю
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
20
31"
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
E0
F1"
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
X0
Y1"
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
k0
l1"
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
~0
1"
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
0
С0
Т1"
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
0
д0
е1"
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
0
╖0
╕1"
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
·2ў
+__inference_Generator_layer_call_fn_3414944
+__inference_Generator_layer_call_fn_3415179
+__inference_Generator_layer_call_fn_3416124
+__inference_Generator_layer_call_fn_3416019└
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
ц2у
F__inference_Generator_layer_call_and_return_conditional_losses_3415608
F__inference_Generator_layer_call_and_return_conditional_losses_3415914
F__inference_Generator_layer_call_and_return_conditional_losses_3414578
F__inference_Generator_layer_call_and_return_conditional_losses_3414708└
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
у2р
"__inference__wrapped_model_3412857╣
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
annotationsк *)в&
$К!
	gen_noise         с
╙2╨
)__inference_reshape_layer_call_fn_3416143в
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
D__inference_reshape_layer_call_and_return_conditional_losses_3416138в
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
З2Д
(__inference_covtr4_layer_call_fn_3412905╫
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
в2Я
C__inference_covtr4_layer_call_and_return_conditional_losses_3412895╫
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
╫2╘
-__inference_leaky_re_lu_layer_call_fn_3416153в
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
Є2я
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_3416148в
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
и2е
5__inference_batch_normalization_layer_call_fn_3416204
5__inference_batch_normalization_layer_call_fn_3416217┤
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
▐2█
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3416191
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3416173┤
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
З2Д
(__inference_covtr5_layer_call_fn_3413057╫
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
2К/+                           
в2Я
C__inference_covtr5_layer_call_and_return_conditional_losses_3413047╫
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
2К/+                           
┘2╓
/__inference_leaky_re_lu_1_layer_call_fn_3416227в
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
Ї2ё
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_3416222в
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
м2й
7__inference_batch_normalization_1_layer_call_fn_3416291
7__inference_batch_normalization_1_layer_call_fn_3416278┤
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
т2▀
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3416247
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3416265┤
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
З2Д
(__inference_covtr6_layer_call_fn_3413209╫
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
2К/+                           
в2Я
C__inference_covtr6_layer_call_and_return_conditional_losses_3413199╫
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
2К/+                           
┘2╓
/__inference_leaky_re_lu_2_layer_call_fn_3416301в
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
Ї2ё
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_3416296в
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
м2й
7__inference_batch_normalization_2_layer_call_fn_3416352
7__inference_batch_normalization_2_layer_call_fn_3416365┤
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
т2▀
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3416339
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3416321┤
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
З2Д
(__inference_covtr7_layer_call_fn_3413361╫
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
2К/+                           
в2Я
C__inference_covtr7_layer_call_and_return_conditional_losses_3413351╫
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
2К/+                           
┘2╓
/__inference_leaky_re_lu_3_layer_call_fn_3416375в
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
Ї2ё
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_3416370в
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
м2й
7__inference_batch_normalization_3_layer_call_fn_3416426
7__inference_batch_normalization_3_layer_call_fn_3416439┤
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
т2▀
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3416413
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3416395┤
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
З2Д
(__inference_covtr8_layer_call_fn_3413513╫
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
2К/+                           
в2Я
C__inference_covtr8_layer_call_and_return_conditional_losses_3413503╫
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
2К/+                           
┘2╓
/__inference_leaky_re_lu_4_layer_call_fn_3416449в
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
Ї2ё
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_3416444в
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
м2й
7__inference_batch_normalization_4_layer_call_fn_3416513
7__inference_batch_normalization_4_layer_call_fn_3416500┤
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
т2▀
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_3416487
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_3416469┤
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
З2Д
(__inference_covtr9_layer_call_fn_3413665╫
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
2К/+                           
в2Я
C__inference_covtr9_layer_call_and_return_conditional_losses_3413655╫
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
2К/+                           
┘2╓
/__inference_leaky_re_lu_5_layer_call_fn_3416523в
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
Ї2ё
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_3416518в
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
м2й
7__inference_batch_normalization_5_layer_call_fn_3416574
7__inference_batch_normalization_5_layer_call_fn_3416587┤
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
т2▀
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_3416561
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_3416543┤
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
И2Е
)__inference_covtr10_layer_call_fn_3413817╫
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
2К/+                           
г2а
D__inference_covtr10_layer_call_and_return_conditional_losses_3413807╫
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
2К/+                           
┘2╓
/__inference_leaky_re_lu_6_layer_call_fn_3416597в
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
Ї2ё
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_3416592в
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
м2й
7__inference_batch_normalization_6_layer_call_fn_3416648
7__inference_batch_normalization_6_layer_call_fn_3416661┤
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
т2▀
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3416635
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3416617┤
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
И2Е
)__inference_covtr11_layer_call_fn_3413969╫
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
2К/+                           
г2а
D__inference_covtr11_layer_call_and_return_conditional_losses_3413959╫
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
2К/+                           
┘2╓
/__inference_leaky_re_lu_7_layer_call_fn_3416671в
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
Ї2ё
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_3416666в
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
м2й
7__inference_batch_normalization_7_layer_call_fn_3416735
7__inference_batch_normalization_7_layer_call_fn_3416722┤
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
т2▀
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3416709
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3416691┤
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
И2Е
)__inference_covtr14_layer_call_fn_3414122╫
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
2К/+                           "
г2а
D__inference_covtr14_layer_call_and_return_conditional_losses_3414112╫
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
2К/+                           "
6B4
%__inference_signature_wrapper_3415286	gen_noiseС
F__inference_Generator_layer_call_and_return_conditional_losses_3414578╞F%&012389CDEFKLVWXY^_ijklqr|}~ДЕПРСТЧШвгдекл╡╢╖╕╜╛;в8
1в.
$К!
	gen_noise         с
p

 
к "?в<
5К2
0+                           
Ъ С
F__inference_Generator_layer_call_and_return_conditional_losses_3414708╞F%&012389CDEFKLVWXY^_ijklqr|}~ДЕПРСТЧШвгдекл╡╢╖╕╜╛;в8
1в.
$К!
	gen_noise         с
p 

 
к "?в<
5К2
0+                           
Ъ №
F__inference_Generator_layer_call_and_return_conditional_losses_3415608▒F%&012389CDEFKLVWXY^_ijklqr|}~ДЕПРСТЧШвгдекл╡╢╖╕╜╛8в5
.в+
!К
inputs         с
p

 
к "-в*
#К 
0         7-
Ъ №
F__inference_Generator_layer_call_and_return_conditional_losses_3415914▒F%&012389CDEFKLVWXY^_ijklqr|}~ДЕПРСТЧШвгдекл╡╢╖╕╜╛8в5
.в+
!К
inputs         с
p 

 
к "-в*
#К 
0         7-
Ъ щ
+__inference_Generator_layer_call_fn_3414944╣F%&012389CDEFKLVWXY^_ijklqr|}~ДЕПРСТЧШвгдекл╡╢╖╕╜╛;в8
1в.
$К!
	gen_noise         с
p

 
к "2К/+                           щ
+__inference_Generator_layer_call_fn_3415179╣F%&012389CDEFKLVWXY^_ijklqr|}~ДЕПРСТЧШвгдекл╡╢╖╕╜╛;в8
1в.
$К!
	gen_noise         с
p 

 
к "2К/+                           ц
+__inference_Generator_layer_call_fn_3416019╢F%&012389CDEFKLVWXY^_ijklqr|}~ДЕПРСТЧШвгдекл╡╢╖╕╜╛8в5
.в+
!К
inputs         с
p

 
к "2К/+                           ц
+__inference_Generator_layer_call_fn_3416124╢F%&012389CDEFKLVWXY^_ijklqr|}~ДЕПРСТЧШвгдекл╡╢╖╕╜╛8в5
.в+
!К
inputs         с
p 

 
к "2К/+                           ▀
"__inference__wrapped_model_3412857╕F%&012389CDEFKLVWXY^_ijklqr|}~ДЕПРСТЧШвгдекл╡╢╖╕╜╛3в0
)в&
$К!
	gen_noise         с
к "9к6
4
covtr14)К&
covtr14         7-э
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3416247ЦCDEFMвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ э
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3416265ЦCDEFMвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ┼
7__inference_batch_normalization_1_layer_call_fn_3416278ЙCDEFMвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ┼
7__inference_batch_normalization_1_layer_call_fn_3416291ЙCDEFMвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           э
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3416321ЦVWXYMвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ э
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3416339ЦVWXYMвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ┼
7__inference_batch_normalization_2_layer_call_fn_3416352ЙVWXYMвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ┼
7__inference_batch_normalization_2_layer_call_fn_3416365ЙVWXYMвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           э
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3416395ЦijklMвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ э
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3416413ЦijklMвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ┼
7__inference_batch_normalization_3_layer_call_fn_3416426ЙijklMвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ┼
7__inference_batch_normalization_3_layer_call_fn_3416439ЙijklMвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           э
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_3416469Ц|}~MвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ э
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_3416487Ц|}~MвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ┼
7__inference_batch_normalization_4_layer_call_fn_3416500Й|}~MвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ┼
7__inference_batch_normalization_4_layer_call_fn_3416513Й|}~MвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           ё
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_3416543ЪПРСТMвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ё
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_3416561ЪПРСТMвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ╔
7__inference_batch_normalization_5_layer_call_fn_3416574НПРСТMвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ╔
7__inference_batch_normalization_5_layer_call_fn_3416587НПРСТMвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           ё
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3416617ЪвгдеMвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ё
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3416635ЪвгдеMвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ╔
7__inference_batch_normalization_6_layer_call_fn_3416648НвгдеMвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ╔
7__inference_batch_normalization_6_layer_call_fn_3416661НвгдеMвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           ё
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3416691Ъ╡╢╖╕MвJ
Cв@
:К7
inputs+                           "
p
к "?в<
5К2
0+                           "
Ъ ё
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3416709Ъ╡╢╖╕MвJ
Cв@
:К7
inputs+                           "
p 
к "?в<
5К2
0+                           "
Ъ ╔
7__inference_batch_normalization_7_layer_call_fn_3416722Н╡╢╖╕MвJ
Cв@
:К7
inputs+                           "
p
к "2К/+                           "╔
7__inference_batch_normalization_7_layer_call_fn_3416735Н╡╢╖╕MвJ
Cв@
:К7
inputs+                           "
p 
к "2К/+                           "ы
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3416173Ц0123MвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ы
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3416191Ц0123MвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ├
5__inference_batch_normalization_layer_call_fn_3416204Й0123MвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ├
5__inference_batch_normalization_layer_call_fn_3416217Й0123MвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           █
D__inference_covtr10_layer_call_and_return_conditional_losses_3413807ТЧШIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ │
)__inference_covtr10_layer_call_fn_3413817ЕЧШIвF
?в<
:К7
inputs+                           
к "2К/+                           █
D__inference_covtr11_layer_call_and_return_conditional_losses_3413959ТклIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           "
Ъ │
)__inference_covtr11_layer_call_fn_3413969ЕклIвF
?в<
:К7
inputs+                           
к "2К/+                           "█
D__inference_covtr14_layer_call_and_return_conditional_losses_3414112Т╜╛IвF
?в<
:К7
inputs+                           "
к "?в<
5К2
0+                           
Ъ │
)__inference_covtr14_layer_call_fn_3414122Е╜╛IвF
?в<
:К7
inputs+                           "
к "2К/+                           ╪
C__inference_covtr4_layer_call_and_return_conditional_losses_3412895Р%&IвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ░
(__inference_covtr4_layer_call_fn_3412905Г%&IвF
?в<
:К7
inputs+                           
к "2К/+                           ╪
C__inference_covtr5_layer_call_and_return_conditional_losses_3413047Р89IвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ░
(__inference_covtr5_layer_call_fn_3413057Г89IвF
?в<
:К7
inputs+                           
к "2К/+                           ╪
C__inference_covtr6_layer_call_and_return_conditional_losses_3413199РKLIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ░
(__inference_covtr6_layer_call_fn_3413209ГKLIвF
?в<
:К7
inputs+                           
к "2К/+                           ╪
C__inference_covtr7_layer_call_and_return_conditional_losses_3413351Р^_IвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ░
(__inference_covtr7_layer_call_fn_3413361Г^_IвF
?в<
:К7
inputs+                           
к "2К/+                           ╪
C__inference_covtr8_layer_call_and_return_conditional_losses_3413503РqrIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ░
(__inference_covtr8_layer_call_fn_3413513ГqrIвF
?в<
:К7
inputs+                           
к "2К/+                           ┌
C__inference_covtr9_layer_call_and_return_conditional_losses_3413655ТДЕIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ▓
(__inference_covtr9_layer_call_fn_3413665ЕДЕIвF
?в<
:К7
inputs+                           
к "2К/+                           █
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_3416222МIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ▓
/__inference_leaky_re_lu_1_layer_call_fn_3416227IвF
?в<
:К7
inputs+                           
к "2К/+                           █
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_3416296МIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ▓
/__inference_leaky_re_lu_2_layer_call_fn_3416301IвF
?в<
:К7
inputs+                           
к "2К/+                           █
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_3416370МIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ▓
/__inference_leaky_re_lu_3_layer_call_fn_3416375IвF
?в<
:К7
inputs+                           
к "2К/+                           █
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_3416444МIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ▓
/__inference_leaky_re_lu_4_layer_call_fn_3416449IвF
?в<
:К7
inputs+                           
к "2К/+                           █
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_3416518МIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ▓
/__inference_leaky_re_lu_5_layer_call_fn_3416523IвF
?в<
:К7
inputs+                           
к "2К/+                           █
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_3416592МIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ▓
/__inference_leaky_re_lu_6_layer_call_fn_3416597IвF
?в<
:К7
inputs+                           
к "2К/+                           █
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_3416666МIвF
?в<
:К7
inputs+                           "
к "?в<
5К2
0+                           "
Ъ ▓
/__inference_leaky_re_lu_7_layer_call_fn_3416671IвF
?в<
:К7
inputs+                           "
к "2К/+                           "┘
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_3416148МIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ░
-__inference_leaky_re_lu_layer_call_fn_3416153IвF
?в<
:К7
inputs+                           
к "2К/+                           й
D__inference_reshape_layer_call_and_return_conditional_losses_3416138a0в-
&в#
!К
inputs         с
к "-в*
#К 
0         
Ъ Б
)__inference_reshape_layer_call_fn_3416143T0в-
&в#
!К
inputs         с
к " К         я
%__inference_signature_wrapper_3415286┼F%&012389CDEFKLVWXY^_ijklqr|}~ДЕПРСТЧШвгдекл╡╢╖╕╜╛@в=
в 
6к3
1
	gen_noise$К!
	gen_noise         с"9к6
4
covtr14)К&
covtr14         7-