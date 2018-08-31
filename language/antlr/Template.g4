grammar Template;

DISTRIBUTION: 'bernoulli'  | 'normal' | 'invgamma' | 'gamma' ;
INTEGERTYPE: 'int';
FLOATTYPE: 'float';
INT : [\-]?[0-9]+;
DOUBLE : [\-]?([0-9]? '.' [0-9]+) | ([1-9][0-9]* '.' [0-9]+) | [0-9]+'E'[0-9]+ ;
DISTHOLE: 'DIST';
CONSTHOLE: 'CONST';
DISTXHOLE: 'DISTX';
AOP: '+' | '-' | '*';
BOP: '>=' | '<=' | '<' | '>' | '==';
WS : [ \n\t\r]+ -> channel(HIDDEN) ;
ID: [a-zA-Z]+[a-zA-Z0-9_]*;

primitive: INTEGERTYPE | FLOATTYPE;
number: INT | DOUBLE | CONSTHOLE;
dims: INT (',' INT)*;
dtype: primitive | (primitive '[' dims ']');
array : '[' number ( ',' number )* ']';
data: ID ':' dtype | ID ':' expr | ID ':' array ;
param: CONSTHOLE | expr;
params: param (',' param )*;
distexpr: (DISTHOLE | DISTRIBUTION) '(' params ')' ('[' dims ']')? | DISTXHOLE ('[' dims ']')?;

expr: number            #val
    | ID                #ref
    | expr AOP expr     #arith
    | expr BOP expr     #comp
    ;

assign: ID '=' distexpr | ID '=' expr;
prior: ID ':=' distexpr  ;
lrmodel: 'observe' '('  distexpr  ',' ID ')';
condmodel: 'if' '(' expr ')' 'then' model 'else' model;
model: lrmodel | condmodel | assign;
query: ('posterior' | 'expectation') '(' ID ')' ;
template : data+ prior+ model* query+;

