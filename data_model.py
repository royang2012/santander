
class Customer:
    def __init__(self, ncodpers, ind_empleado, pais_residencia, sexo, age,
                 contract_history, antiguedad, indrel, tiprel_1mes, indresi,
                 conyuemp, canal_entrada, indfall, cod_prov, ind_actividad_cliente,
                 ):
        self.customer_code = ncodpers
        self.employee = ind_empleado
        self.resident_country = pais_residencia
        self.gender = sexo
        self.age = age
        self.contract_history = contract_history
        self.seniority = antiguedad
        self.primary = indrel
        self.relation_type = tiprel_1mes
        self.is_residence = indresi
        self.spouse_index = conyuemp
        self.channel_join = canal_entrada
        self.deceased_index = indfall
        self.provice = cod_prov
        self.is_active = ind_actividad_cliente
