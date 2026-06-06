open Printf
(* module Gp = Gnuplot *)

let string_includes s sub =
    let len_s = String.length s in
    let len_sub = String.length sub in
    let rec aux i =
        if i + len_sub > len_s then false
        else if String.sub s i len_sub = sub then true
        else aux (i + 1)
    in
    aux 0

let read_csv filename =
    let ic = open_in filename in
    let header = input_line ic in
    (* skip header *)

    let ch1 = ref [] in
    let ch2 = ref [] in

    try
      while true do
        let line = input_line ic in
        let parts = String.split_on_char ',' line in
        match parts with
        | v1 :: v2 :: _ ->
            ch1 := float_of_string v1 :: !ch1;
            ch2 := float_of_string v2 :: !ch2
        | _ -> ()
      done;
      assert false
    with End_of_file ->
      close_in ic;
      (Array.of_list (List.rev !ch1), Array.of_list (List.rev !ch2), header)

let write_csv filename header ch1 ch2 =
    let oc = open_out filename in
    fprintf oc "%s\n" header;

    for i = 0 to Array.length ch1 - 1 do
      fprintf oc "%f,%f\n" ch1.(i) ch2.(i)
    done;

    close_out oc

let getTimeInfo header =
    (* String.split_on_char ',' header in *)
    let t0 = ref 0.0 in
    let tInc = ref 0.0 in
    let rec aux parts =
        match parts with
        | part :: rest ->
            (match String.split_on_char '=' part with
            | label :: value :: _ when string_includes label "t0" ->
                t0 := float_of_string value
            | label :: value :: _ when string_includes label "tInc" ->
                tInc := float_of_string value
            | _ -> ());
            aux rest
        | [] -> ()
    in
    aux (String.split_on_char ',' header);
    (!t0, !tInc)


let normalize_at_frequency x fs f0 =
  let n = Array.length x in
  let y = Array.make n 0.0 in
  let omega = 2.0 *. Float.pi *. f0 in

  (* I and Q components *)
  let i_comp = Array.make n 0.0 in
  let q_comp = Array.make n 0.0 in

  for k = 0 to n - 1 do
    let t = float_of_int k /. fs in
    i_comp.(k) <- x.(k) *. cos (omega *. t);
    q_comp.(k) <- x.(k) *. sin (omega *. t);
  done;

  (* Simple low-pass via moving average *)
  let window = 200 in
  let half = window / 2 in

  let envelope = Array.make n 1.0 in

  for k = 0 to n - 1 do
    let start_i = max 0 (k - half) in
    let end_i = min n (k + half) in
    let sum_i = ref 0.0 in
    let sum_q = ref 0.0 in
    let count = ref 0 in

    for j = start_i to end_i - 1 do
      sum_i := !sum_i +. i_comp.(j);
      sum_q := !sum_q +. q_comp.(j);
      incr count
    done;

    let i_avg = !sum_i /. float_of_int !count in
    let q_avg = !sum_q /. float_of_int !count in
    envelope.(k) <- sqrt (i_avg *. i_avg +. q_avg *. q_avg)
  done;

  (* Normalize *)
  for k = 0 to n - 1 do
    if envelope.(k) > 1e-12 then
      y.(k) <- x.(k) /. envelope.(k)
    else
      y.(k) <- 0.0
  done;

  y




let () = 
    let ch1, ch2, header = read_csv "./RigolDS0.csv" in
    let t0, tInc = getTimeInfo header in

    printf "t0=%f   tInc=%f\n" t0 tInc;
    (* let n = Array.length ch1 in *)
    (* let t = Array.init n (fun i -> t0 +. (float_of_int i *. tInc)) in *)

    (* Array.iter (Printf.printf "%f,") ch1; print_newline (); () *)

    let ret = normalize_at_frequency ch1 35.4 35.4 in

    write_csv "out.csv" header ret ch2