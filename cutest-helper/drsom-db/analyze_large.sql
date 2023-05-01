WITH ranked_messages AS (SELECT m.*, ROW_NUMBER() OVER (PARTITION BY name,param,method ORDER BY `update` DESC) AS rn
                         FROM cutest.result AS m)
select t.method,
       t.nf,
       t.tf,
       t.kf,
       t.kff,
       t.kfg,
       t.kfh,
       tt.tg,
       tt.kg,
       tt.kgf,
       tt.kgg,
       tt.kgh,
       tt.version
from (select method,
             rn          as version,
             sum(status) as nf,
             avg(t)      as tf,
             avg(k)      as kf,
             avg(kf)     as kff,
             avg(kg)     as kfg,
             avg(kh)     as kfh
      from ranked_messages
      where k <= 50000
        and t <= 1000
#         and n <= 200
        and `precision` = 1e-5
      group by method, rn)
         as t
         left join (select method,
                           rn                         as version,
                           exp(avg(ln(t + 1))) - 1    as tg,
                           exp(avg(ln(k + 50))) - 50  as kg,
                           exp(avg(ln(kf + 50))) - 50 as kgf,
                           exp(avg(ln(kg + 50))) - 50 as kgg,
                           exp(avg(ln(kh + 50))) - 50 as kgh
                    from ranked_messages
                    group by method, rn) as tt
                   on tt.method = t.method and tt.version = t.version
where
    tt.version <= 1
    and tt.method not in ('\\cg', '\\gd', '\\drsom', '\\lbfgs');
# and    tt.method not in ('\\newtontr', '\\hsodm', '\\arc');