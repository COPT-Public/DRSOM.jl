WITH ranked_messages AS (SELECT m.*, ROW_NUMBER() OVER (PARTITION BY name,param,method ORDER BY `update` DESC) AS rn
                         FROM cutest.result AS m)
select t.method,
       t.up_max,
       t.up_min,
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
             max(`update`)                 as up_max,
             min(`update`)                 as up_min,
             rn                            as version,
             sum(status)                   as nf,
             avg(if(status = 0, 20000, t))   as tf,
             avg(if(status = 0, 20000, k)) as kf,
             avg(kf)                       as kff,
             avg(kg)                       as kfg,
             avg(kh)                       as kfh
      from ranked_messages
      where `precision` = 1e-5
      group by method, rn)
         as t
         left join (select method,
                           rn                                               as version,
                           exp(avg(ln(if(status = 0, 20000, t) + 1))) - 1     as tg,
                           exp(avg(ln(if(status = 0, 20000, k) + 50))) - 50 as kg,
                           exp(avg(ln(kf + 50))) - 50                       as kgf,
                           exp(avg(ln(kg + 50))) - 50                       as kgg,
                           exp(avg(ln(kh + 50))) - 50                       as kgh
                    from ranked_messages
                    where `precision` = 1e-5
                    group by method, rn) as tt
                   on tt.method = t.method and tt.version = t.version
where tt.version <= 2
  and tt.method not in ('\\lbfgs', '\\cg', '\\gd', '\\drsom');